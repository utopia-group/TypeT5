import multiprocessing

from datasets import Dataset
from spot.critic import CriticCollator, CriticModel

from spot.data import (
    _TokenizedSrcHelper,
    ChunkedDataset,
    CtxArgs,
    SrcCheckResult,
    SrcChunkInfo,
    SrcDataset,
    TokenizedSrc,
    code_to_check_from_preds,
    src_to_chunks_,
    type_check_src_in_project,
)
from spot.model import DatasetPredResult, DecodingArgs, ModelWrapper, dynamic_dataloader
from spot.type_check import (
    MypyChecker,
    MypyFeedback,
    MypyResult,
    PythonType,
    normalize_type,
)
from spot.utils import *


def sample_candidates(
    wrapper: ModelWrapper,
    src_data: SrcDataset,
    n_samples: int,
) -> tuple[ChunkedDataset, list[list[list[PythonType]]]]:
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

    pred_candidates = [
        [get_preds(cid, sid) for sid in range(n_samples)] for cid in range(n_chunks)
    ]  # of shape (n_chunks, n_samples, n_labels)
    return chunks, pred_candidates


def select_candidates_by_type_errors(
    src_data: SrcDataset,
    chunks: ChunkedDataset,
    pred_candidates: list[list[list[PythonType]]],
) -> DatasetPredResult:
    file2src = src_data.file2src(resolve=False)
    srcs_to_check = src_data.srcs_with_labels()

    with src_data.prepare_typecheck_projects(srcs_to_check) as template_root:

        to_check = dict[tuple[int, int], tuple[TokenizedSrc, dict[int, str], Path]]()
        for i in range(len(chunks.data)):
            info = chunks.chunks_info[i]
            file = chunks.files[info.src_ids[0]]
            src = file2src[file.relative_to(src_data.repos_root)]
            proj_root = template_root / src.repo
            for j, candidates in enumerate(pred_candidates[i]):
                preds_dict = {
                    l_id: str(pred) for l_id, pred in zip(info.label_ids, candidates)
                }
                to_check[(i, j)] = (src, preds_dict, proj_root)

        to_check_values = to_check.values()
        check_rs: list[int] = pmap(
            count_type_errors_in_project,
            [[x[0]] for x in to_check_values],
            [[x[1]] for x in to_check_values],
            [x[2] for x in to_check_values],
            desc="map count_type_errors_in_project",
        )

    n_errors = dict(zip(to_check.keys(), check_rs))
    final_preds = list[list[PythonType]]()
    extra_info = list[dict]()
    for i in range(len(chunks.data)):
        candidates = pred_candidates[i]
        es = [n_errors[(i, j)] for j in range(len(candidates))]
        sample_id = int(np.argmin(es))
        final_preds.append(candidates[sample_id])
        extra_info.append({"n_errors": es[sample_id]})
    return DatasetPredResult(chunks, final_preds, extra_info)


def select_candidates_using_oracle(
    chunks: ChunkedDataset,
    pred_candidates: list[list[list[PythonType]]],
) -> DatasetPredResult:
    final_preds = list[list[PythonType]]()
    extra_info = list[dict]()
    for i in range(len(chunks.data)):
        info = chunks.chunks_info[i]
        candidates = pred_candidates[i]
        n_errors = []
        for preds in candidates:
            ne = sum(
                0 if normalize_type(p) == normalize_type(t) else 1
                for p, t in zip(preds, info.types)
            )
            n_errors.append(ne)
        sample_id = int(np.argmin(n_errors))
        final_preds.append(candidates[sample_id])
        extra_info.append({"n_errors_oracle": n_errors[sample_id]})

    return DatasetPredResult(chunks, final_preds, extra_info)


def select_candidates_using_critic(
    critic: CriticModel,
    src_data: SrcDataset,
    chunks: ChunkedDataset,
    pred_candidates: list[list[list[PythonType]]],
    dec_args: DecodingArgs,
    tqdm_args: dict = {},
    use_logp: bool = False,
) -> DatasetPredResult:
    file2src = src_data.file2src(resolve=False)
    srcs_to_check = src_data.srcs_with_labels()

    with src_data.prepare_typecheck_projects(srcs_to_check) as template_root:

        to_check = dict[tuple[int, int], tuple[TokenizedSrc, dict[int, str], Path]]()
        for i in range(len(chunks.data)):
            info = chunks.chunks_info[i]
            file = chunks.files[info.src_ids[0]]
            src = file2src[file.relative_to(src_data.repos_root)]
            proj_root = template_root / src.repo
            for j, candidates in enumerate(pred_candidates[i]):
                preds_dict = {
                    l_id: str(pred) for l_id, pred in zip(info.label_ids, candidates)
                }
                to_check[(i, j)] = (src, preds_dict, proj_root)

        to_check_values = to_check.values()
        check_rs: list[SrcCheckResult] = pmap(
            type_check_src_in_project,
            [x[0] for x in to_check_values],
            [x[1] for x in to_check_values],
            [x[2] for x in to_check_values],
            desc="map type_check_src_in_project",
            tqdm_args=tqdm_args,
        )

    all_fdbks = [r.feedbacks for r in check_rs if isinstance(r.feedbacks, list)]
    success_rate = len(all_fdbks) / len(check_rs)
    print(f"Type checking success rate: {success_rate:.2%}")

    n_errors_map = dict[tuple[int, int], int]()
    for k, v in zip(to_check.keys(), [r.feedbacks for r in check_rs]):
        if isinstance(v, list):
            n_errors_map[k] = len(v)
        else:
            n_errors_map[k] = 0

    avg_n_fdbks = sum(n_errors_map.values()) / len(n_errors_map)
    print(f"Average number of feedbacks per check: {avg_n_fdbks:.2f}")

    file2id = src_data.file2id()
    src_ids = [file2id[x[0].file] for x in to_check_values]
    critic_inputs_metas = pmap(
        to_critic_inputs,
        [x[0] for x in to_check_values],
        src_ids,
        [x[1] for x in to_check_values],
        check_rs,
        [dec_args.ctx_args] * len(to_check_values),
        desc="map to_critic_inputs",
    )
    all_inputs = [x for xs, _ in critic_inputs_metas for x in xs]
    all_meta = [x for _, xs in critic_inputs_metas for x in xs]
    for i, (x, info) in enumerate(zip(all_inputs, all_meta)):
        x["chunk_id"] = i
        x["prediction_spans"] = [
            (s.start, s.stop) for s in not_none(info.inlined_spans)
        ]

    critic_dataset = Dataset.from_dict(merge_dicts(all_inputs))

    dataloader = dynamic_dataloader(
        critic_dataset,
        max_tokens=dec_args.sampling_max_tokens,
        collate_fn=CriticCollator(DefaultTokenizer),
    )
    chunk2preds = critic.classify_data(dataloader, len(all_inputs), tqdm_args=tqdm_args)
    # the number of correct predictions judged by the critic
    critic_preds = list[list[float]]()
    critic_scores = list[float]()
    for inputs, metas in critic_inputs_metas:
        all_preds = []
        for x, meta in zip(inputs, metas):
            preds = chunk2preds[x["chunk_id"]]
            assert_eq(len(preds), len(not_none(meta.prev_types)))
            all_preds.extend(preds)
        critic_preds.append(all_preds)
        score = (
            np.mean([math.log(p) for p in all_preds])
            if use_logp
            else np.mean(all_preds)
        )
        critic_scores.append(score)

    scores_map = dict(zip(to_check.keys(), critic_scores))
    preds_map = dict(zip(to_check.keys(), critic_preds))
    final_preds = list[list[PythonType]]()
    extra_info = list[dict]()
    final_errors = 0
    for i in range(len(chunks.data)):
        candidates = pred_candidates[i]
        cand_scores = [scores_map[(i, j)] for j in range(len(candidates))]
        sample_id = int(np.argmax(cand_scores))
        final_errors += n_errors_map[(i, sample_id)]
        final_preds.append(candidates[sample_id])
        extra_info.append(
            {
                "critic_score": cand_scores[sample_id],
                "critic_preds": preds_map[(i, sample_id)],
            }
        )
    print("Average number of errors after selection:", final_errors / len(chunks.data))
    return DatasetPredResult(chunks, final_preds, extra_info)


def to_critic_inputs(
    src: TokenizedSrc,
    src_id: int,
    preds: dict[int, PythonType],
    check_r: SrcCheckResult,
    ctx_args: CtxArgs,
):
    """
    Patch each src with the type checker feedbacks and inline the previous predicitons,
    then break the src into one (if short enough) or more chunks.
    """
    errors, current_code = check_r
    fdbks = [] if isinstance(errors, str) else errors
    helper = _TokenizedSrcHelper(DefaultTokenizer)
    new_src = helper.feedbacks_to_tokenized_src(
        src, current_code, fdbks, patch_predictions=False
    )
    new_src.prev_types = preds
    new_src = TokenizedSrc.inline_predictions(new_src)
    chunks = list[dict]()
    chunks_info = list[SrcChunkInfo]()
    labels_range = min(preds.keys()), max(preds.keys()) + 1
    src_to_chunks_(
        chunks, chunks_info, new_src, src_id, labels_range, ctx_args, DefaultTokenizer
    )
    return chunks, chunks_info


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
    srcs: list[TokenizedSrc],
    preds_list: list[dict[int, str]],
    proj_root: Path,
    only_errors_in_srcs: bool = False,
) -> int:
    r = collect_type_errors_in_project(srcs, preds_list, proj_root)
    if isinstance(r, MypyResult):
        if only_errors_in_srcs:
            file_errors = [
                r.error_dict.get(s.file.relative_to(s.repo), []) for s in srcs
            ]
        else:
            file_errors = list(r.error_dict.values())
        return sum(len(ls) for ls in file_errors)
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
    cwd.mkdir(parents=True, exist_ok=True)

    for f in project_root.glob("**/*.py"):
        rel_path = f.relative_to(project_root)
        (cwd / rel_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, cwd / rel_path)

    for src, preds in zip(srcs, preds_list):
        rel_path = src.file.relative_to(src.repo)
        file_path = cwd / rel_path
        new_code = code_to_check_from_preds(src, preds)
        file_path.write_text(new_code)
    check_r = MypyChecker.check_project(cwd, mypy_path=mypy_path)
    if isinstance(check_r, MypyResult):
        check_r.error_dict = {
            f.relative_to(cwd): es for f, es in check_r.error_dict.items()
        }
    return check_r
