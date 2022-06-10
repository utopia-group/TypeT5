import multiprocessing
from shutil import rmtree

from spot.data import (
    SrcCheckResult,
    SrcDataset,
    TokenizedSrc,
    code_to_check_from_preds,
    type_check_src_in_project,
)
from spot.model import DatasetPredResult, ModelWrapper
from spot.type_check import MypyChecker, MypyFeedback, MypyResult, PythonType
from spot.utils import *


def sample_then_select(
    wrapper: ModelWrapper, src_data: SrcDataset, n_samples: int
) -> DatasetPredResult:
    """
    Sample the solutions for each file using top-p sampling or (diverse) beam search
    and then select the best solution according to the type checker feedbacks.
    """
    tokenizer = wrapper.tokenizer
    ctx_args = wrapper.args.ctx_args

    do_sample = wrapper.args.do_sample
    if not do_sample:
        assert wrapper.args.num_beams is not None, "num_beams needs to be set"
        assert n_samples <= wrapper.args.num_beams

    chunks = src_data.to_chunks(tokenizer, ctx_args, tqdm_args={"disable": True})
    n_chunks = len(chunks.data)

    if do_sample:
        samples = [
            wrapper.predict(chunks.data, tqdm_args={"leave": False})
            for _ in tqdm(range(n_samples), desc="Sampling")
        ]  # of shape (n_samples, n_chunks, n_labels)
    else:
        samples = wrapper.predict(
            chunks.data,
            num_return_sequences=n_samples,
            tqdm_args={},
        )  # of shape (n_chunks, n_samples, n_labels)
        assert_eq(len(samples), n_chunks)
        assert_eq(len(samples[0]), n_samples)

    def get_preds(chunk_id, sample_id):
        return (
            samples[sample_id][chunk_id] if do_sample else samples[chunk_id][sample_id]
        )

    file2src = src_data.file2src(resolve=False)

    srcs_to_check = src_data.srcs_with_labels()
    with src_data.prepare_typecheck_projects(srcs_to_check) as template_root:

        to_check = dict[tuple[int, int], tuple[TokenizedSrc, dict[int, str], Path]]()
        for i in range(n_chunks):
            info = chunks.chunks_info[i]
            file = chunks.files[info.src_ids[0]]
            src = file2src[file.relative_to(src_data.repos_root)]
            proj_root = template_root / src.repo
            for j in range(n_samples):
                preds = get_preds(i, j)
                preds_dict = {
                    l_id: str(pred) for l_id, pred in zip(info.label_ids, preds)
                }
                to_check[(i, j)] = (src, preds_dict, proj_root)

        to_check_values = to_check.values()
        max_workers = wrapper.args.max_workers
        check_rs: list[int] = process_map(
            count_type_errors_in_project,
            [[x[0]] for x in to_check_values],
            [[x[1]] for x in to_check_values],
            [x[2] for x in to_check_values],
            max_workers=max_workers,
            desc="map type_check_src_in_project",
            chunksize=max(1, len(to_check_values) // (8 * max_workers)),
        )

    check_rs_dict = dict(zip(to_check.keys(), check_rs))

    final_preds = list[list[PythonType]]()
    for i in range(n_chunks):
        errors = [check_rs_dict[(i, j)] for j in range(n_samples)]
        sample_id = int(np.argmin(errors))
        final_preds.append(get_preds(i, sample_id))
    return DatasetPredResult(chunks, final_preds)


def collect_type_errors_from_predictions(
    src_data: SrcDataset, result: DatasetPredResult, max_workers: int
) -> list[tuple[Path, MypyFeedback]]:
    "Apply all the predictioins and call the type checker once per project."

    chunks = result.chunks
    chunks_info = chunks.chunks_info
    chunk_preds = result.predictions

    file2src = src_data.file2src(resolve=False)
    srcs_to_check = src_data.srcs_with_labels()

    with src_data.prepare_typecheck_projects(srcs_to_check) as template_root:
        to_check = dict[Path, dict[Path, dict[int, str]]]()
        for i in range(len(chunks_info)):
            info = chunks.chunks_info[i]
            file = chunks.files[info.src_ids[0]]
            src = file2src[file.relative_to(src_data.repos_root)]
            file = src.file
            proj_root = template_root / src.repo
            if proj_root not in to_check:
                to_check[proj_root] = dict()
            if file not in to_check[proj_root]:
                to_check[proj_root][file] = dict()
            pred_dict = to_check[proj_root][file]
            for l_id, pred in zip(info.label_ids, chunk_preds[i]):
                pred_dict[l_id] = str(pred)

        check_rs: list[MypyResult | str] = process_map(
            collect_type_errors_in_project,
            [[file2src[f] for f in d.keys()] for d in to_check.values()],
            [[preds for preds in d.values()] for d in to_check.values()],
            [root for root in to_check.keys()],
            max_workers=max_workers,
            desc="map type_check_src_in_project",
            chunksize=max(1, len(to_check) // (8 * max_workers)),
            leave=False,
        )
        feebacks = [
            (f, e)
            for x in check_rs
            if isinstance(x, MypyResult)
            for f, ls in x.error_dict.items()
            for e in ls
        ]
    return feebacks


def count_type_errors_in_project(
    srcs: list[TokenizedSrc], preds_list: list[dict[int, str]], proj_root: Path
) -> int:
    r = collect_type_errors_in_project(srcs, preds_list, proj_root)
    if isinstance(r, MypyResult):
        return sum(len(ls) for ls in r.error_dict.values())
    else:
        return 0


def collect_type_errors_in_project(
    srcs: list[TokenizedSrc],
    preds_list: list[dict[int, str]],
    project_root: Path,
    mypy_path: Optional[Path] = None,
) -> MypyResult | str:
    # setup: copy all files into cwd
    proc = multiprocessing.current_process()
    cwd = MypyChecker.temp_dir() / proc.name / project_root.name
    if cwd.exists():
        shutil.rmtree(cwd)
    cwd.mkdir(parents=True, exist_ok=True)

    for f in project_root.glob("**/*.py"):
        rel_path = f.relative_to(project_root)
        (cwd / rel_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, cwd / rel_path)

    try:
        for src, preds in zip(srcs, preds_list):
            rel_path = src.file.relative_to(src.repo)
            file_path = cwd / rel_path
            new_code = code_to_check_from_preds(src, preds)
            file_path.write_text(new_code)
        check_r = MypyChecker.check_project(cwd, mypy_path=mypy_path)
        return check_r
    finally:
        shutil.rmtree(MypyChecker.temp_dir() / proc.name)
