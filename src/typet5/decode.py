import multiprocessing
from copy import deepcopy

import torch
from datasets import Dataset

from .critic import CriticCollator, CriticModel
from .data import (
    ChunkedDataset,
    CtxArgs,
    SrcCheckResult,
    SrcChunkInfo,
    TokenizedSrc,
    TokenizedSrcSet,
    TypeCheckingEnv,
    chunk_from_src,
    code_to_check_from_preds,
    feedbacks_to_tokenized_src,
    src_to_chunks_,
    type_check_src_in_project,
)
from .model import DatasetPredResult, DecodingArgs, ModelWrapper, dynamic_dataloader
from .type_check import (
    MypyChecker,
    MypyFeedback,
    MypyResult,
    PythonType,
    normalize_type,
)
from .utils import *


class IncrSelector:
    "A strategy for selecting the best candidate time at each time step."
    pass


class SelectByOracle(IncrSelector):
    "Select the first candidate that matches the ground truth (if any)."
    pass


class SelectByCounting(IncrSelector):
    "Select the first candidate that has the least type errors."
    pass


@dataclass
class SelectByCritic(IncrSelector):
    "Select the best candidate according to the critic's evaluation."
    critic: CriticModel
    score_transform: Callable[[float], float] = lambda x: x


def incr_inference_with_feedback(
    wrapper: ModelWrapper,
    src_data: TokenizedSrcSet,
    beam_width: int,
    selector: IncrSelector,
    print_times: bool = True,
    log_to: Path | None = None,
):
    """
    Perform incremental inference, one type at a time.
    The best type at each time step is selected using the strategy specified
    by `selector`.
    """
    ctx_args = wrapper.args.ctx_args
    sampling_max_tokens = wrapper.args.sampling_max_tokens
    srcs_to_check = src_data.all_srcs
    if log_to is not None:
        shutil.rmtree(log_to, ignore_errors=True)
        log_to.mkdir(parents=True)

    def updated_dict(d, k, v):
        d = deepcopy(d)
        d[k] = v
        return d

    def maybe_log(src_id, lid, name, fn) -> None:
        if log_to is None or lid >= 10:
            return
        with (log_to / f"src-{src_id}.txt").open("a") as file:
            file.write(f"<========== (label_id={lid}) {name} ==========>\n")
            content = fn()
            content = content if isinstance(content, str) else str(content)
            file.write(content)
            file.write("\n")

    def pretty_show_predictions(types, types_tensor):
        tokens = [
            decode_tokens(tks, skip_special_tokens=True)
            for tks in types_tensor.tolist()
        ]
        return "\n".join([f"{types[i]}     ({tokens[i]})" for i in range(len(types))])

    def infer_src(
        src: TokenizedSrc,
        src_id: int,
        executor: ProcessPoolExecutor,
        env: TypeCheckingEnv,
        prog: tqdm,
    ):
        device = wrapper.model.device
        num_return_sequences = beam_width
        proj_root = env.template_root / src.repo
        assignment = dict[int, PythonType]()
        for lid in range(len(src.types_info)):
            with t_logger.timed("chunk_from_src"):
                chunk, info = chunk_from_src(src, lid, ctx_args)
                batch = dict()
                batch["input_ids"] = torch.tensor([chunk["input_ids"]], device=device)
                batch["n_labels"] = [chunk["n_labels"]]
            # get the list of types return by beam search
            with t_logger.timed("predict_on_batch"):
                maybe_log(
                    src_id, lid, "input_ids", lambda: decode_tokens(chunk["input_ids"])
                )
                types, types_tensor = wrapper.predict_on_batch(
                    batch, num_return_sequences
                )
                preds = list(seq_flatten(types))
                unique_ids = get_unique_ids(preds)
                preds = [preds[i] for i in unique_ids]
                types_tensor = types_tensor[unique_ids, :]
                maybe_log(
                    src_id,
                    lid,
                    "predictions",
                    lambda: pretty_show_predictions(preds, types_tensor),
                )
                new_assignments = [updated_dict(assignment, lid, t) for t in preds]
                N = len(new_assignments)

            # now try each of them and use the one that worked best
            if isinstance(selector, SelectByCounting):
                # count the number of type errors
                with t_logger.timed("count_type_errors_in_project"):
                    check_rs = executor.map(
                        count_type_errors_in_project,
                        [[src]] * N,
                        [[a] for a in new_assignments],
                        [proj_root] * N,
                    )
                    check_rs = list(check_rs)
                    maybe_log(src_id, lid, "n_errors", lambda: check_rs)
                    best_i = int(np.argmin(check_rs))
            elif isinstance(selector, SelectByOracle):
                truth = normalize_type(src.types[lid])
                maybe_log(src_id, lid, "truth (normalized)", lambda: truth)
                normal_options = [normalize_type(a[lid]) for a in new_assignments]
                maybe_log(src_id, lid, "options (normalized)", lambda: normal_options)
                is_correct = [opt == truth for opt in normal_options]
                maybe_log(src_id, lid, "is_correct", lambda: is_correct)
                best_i = int(np.argmax(is_correct))
            elif isinstance(selector, SelectByCritic):
                # use the one with the highest critic score
                with t_logger.timed("type_check_src_in_project"):
                    preexisting = env.pre_fdbks[src.file]
                    check_rs = executor.map(
                        type_check_src_in_project,
                        [src] * N,
                        [a for a in new_assignments],
                        [proj_root] * N,
                        [preexisting] * N,
                    )
                    check_rs = list(check_rs)
                with t_logger.timed("Running critic"):
                    critic_inputs_metas = executor.map(
                        to_critic_inputs,
                        [src] * N,
                        new_assignments,
                        check_rs,
                        [ctx_args] * N,
                        [(lid, lid + 1)] * N,
                    )
                    critic_inputs_metas = list(critic_inputs_metas)
                    # there should only be one chunk for each src
                    all_inputs = [get_single(xs) for xs, _ in critic_inputs_metas]
                    all_meta = [get_single(xs) for _, xs in critic_inputs_metas]
                    assert_eq(len(all_inputs), len(all_meta), N)
                    maybe_log(
                        src_id,
                        lid,
                        "critic_inputs[0]",
                        lambda: decode_tokens(all_inputs[0]["input_ids"]),
                    )
                    maybe_log(
                        src_id,
                        lid,
                        "critic_inputs[-1]",
                        lambda: decode_tokens(all_inputs[-1]["input_ids"]),
                    )
                    for i, (x, info) in enumerate(zip(all_inputs, all_meta)):
                        x["chunk_id"] = i
                        x["prediction_spans"] = [
                            (s.start, s.stop) for s in not_none(info.inlined_spans)
                        ]
                    critic_dataset = Dataset.from_dict(merge_dicts(all_inputs))
                    dataloader = dynamic_dataloader(
                        critic_dataset,
                        max_tokens=sampling_max_tokens,
                        collate_fn=CriticCollator(),
                    )
                    chunk2preds = selector.critic.classify_data(
                        dataloader, len(critic_dataset), tqdm_args={"disable": True}
                    )
                    scores = [get_single(chunk2preds[i]) for i in range(N)]
                    maybe_log(src_id, lid, "scores", lambda: scores)
                    scores = [selector.score_transform(s) for s in scores]
                    best_i = int(np.argmax(scores))
            else:
                raise NotImplementedError(f"Unknown selector type: {type(selector)}")

            assignment = new_assignments[best_i]
            src = inline_single_prediction(src, lid, preds[best_i], as_comment=False)
            prog.update()
        return src, list(assignment.values())

    t_logger = TimeLogger()
    n_workers = min(wrapper.args.max_workers, beam_width)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        compute_pre_fdbks = isinstance(selector, SelectByCritic)
        with src_data.setup_typechecking(
            srcs_to_check, skip_pre_fdbks=not compute_pre_fdbks
        ) as env:
            with tqdm(
                total=sum(len(s.types_info) for s in srcs_to_check),
                desc=f"incr_inference [{type(selector).__name__}]",
            ) as prog:
                srcs = list[TokenizedSrc]()
                predictions = list[list[PythonType]]()
                for i, src in enumerate(srcs_to_check):
                    src, preds = infer_src(src, i, executor, env, prog)
                    srcs.append(src)
                    predictions.append(preds)

    if print_times:
        display(t_logger.as_dataframe())
    return srcs, predictions


def inline_single_prediction(
    src: TokenizedSrc, label_id: int, ty: PythonType, as_comment: bool
) -> "TokenizedSrc":
    tokenizer = DefaultTokenizer
    mask_id = tokenizer.mask_token_id
    to_insert = tokenizer.encode(str(ty), add_special_tokens=False)
    if as_comment:
        comment_start = tokenizer.encode("/* ", add_special_tokens=False)
        comment_end = tokenizer.encode(" */", add_special_tokens=False)
        to_insert = comment_start + to_insert + comment_end

    l_pos = src.types_pos[label_id]
    assert_eq(src.tokenized_code[l_pos], mask_id)

    new_code = src.tokenized_code[:l_pos] + to_insert + src.tokenized_code[l_pos + 1 :]
    # inlined_span = slice(l_pos, l_pos + len(to_insert))
    offset = len(to_insert) - 1
    new_types_pos = [
        pos + offset if i > label_id else pos for i, pos in enumerate(src.types_pos)
    ]

    return TokenizedSrc(
        file=src.file,
        repo=src.repo,
        types=src.types,
        types_pos=new_types_pos,
        types_str=src.types_str,
        types_tks=src.types_tks,
        types_info=src.types_info,
        main_code=src.main_code,
        tokenized_code=new_code,
        preamble_code=src.preamble_code,
        tokenized_preamble=src.tokenized_preamble,
        prev_types=None,  # don't need them for now
        inlined_spans=None,  # don't need them for now
        feedbacks=src.feedbacks,
    )


def sample_candidates(
    wrapper: ModelWrapper,
    src_data: TokenizedSrcSet,
    n_samples: int,
) -> tuple[ChunkedDataset, list[list[list[PythonType]]]]:
    ctx_args = wrapper.args.ctx_args

    do_sample = wrapper.args.do_sample
    if not do_sample:
        assert wrapper.args.num_beams is not None, "num_beams needs to be set"
        assert n_samples <= wrapper.args.num_beams

    chunks = src_data.to_chunks(ctx_args)
    n_chunks = len(chunks.data)

    if do_sample:
        samples = [
            wrapper.predict(chunks.data, tqdm_args={})
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
    src_data: TokenizedSrcSet,
    chunks: ChunkedDataset,
    pred_candidates: list[list[list[PythonType]]],
    only_same_file_error: bool = False,
) -> DatasetPredResult:
    file2src = src_data.file2src(resolve=False)
    srcs_to_check = src_data.all_srcs

    with src_data.setup_typechecking(srcs_to_check, skip_pre_fdbks=True) as env:
        to_check = dict[tuple[int, int], tuple[TokenizedSrc, dict[int, str], Path]]()
        for i in range(len(chunks.data)):
            info = chunks.chunks_info[i]
            file = info.src_file
            src = file2src[file.relative_to(src_data.repos_root)]
            proj_root = env.template_root / src.repo
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
            [only_same_file_error for _ in to_check_values],
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
    for i in tqdm(range(len(chunks.data)), desc="select_candidates_using_oracle"):
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


def select_first_candidates(
    chunks: ChunkedDataset,
    pred_candidates: list[list[list[PythonType]]],
) -> DatasetPredResult[None]:
    final_preds = list[list[PythonType]]()
    for i in range(len(chunks.data)):
        preds = pred_candidates[i][0]
        final_preds.append(preds)

    return DatasetPredResult(chunks, final_preds, [])


@dataclass
class CriticAssesInfo:
    candidate_scores: list[float]
    candidate_label_scores: list[list[float]]

    @property
    def best_candidate(self) -> int:
        return int(np.argmax(self.candidate_scores))


def select_candidates_using_critic(
    critic: CriticModel,
    no_feedback: bool,
    src_data: TokenizedSrcSet,
    chunks: ChunkedDataset,
    pred_candidates: list[list[list[PythonType]]],
    dec_args: DecodingArgs,
    tqdm_args: dict = {},
    score_transform: Callable[[float], float] = lambda x: x,
) -> DatasetPredResult[CriticAssesInfo]:
    file2src = src_data.file2src(resolve=False)
    srcs_to_check = src_data.all_srcs

    with src_data.setup_typechecking(srcs_to_check) as env:
        to_check = dict[tuple[int, int], tuple[TokenizedSrc, dict[int, str], Path]]()
        for i in range(len(chunks.data)):
            info = chunks.chunks_info[i]
            file = info.src_file
            src = file2src[file.relative_to(src_data.repos_root)]
            proj_root = env.template_root / src.repo
            for j, candidates in enumerate(pred_candidates[i]):
                preds_dict = {
                    l_id: str(pred) for l_id, pred in zip(info.label_ids, candidates)
                }
                to_check[(i, j)] = (src, preds_dict, proj_root)

        to_check_values = to_check.values()
        if no_feedback:
            check_rs = [
                SrcCheckResult("no_feedback=True", x[0].main_code)
                for x in to_check_values
            ]
        else:
            check_rs: list[SrcCheckResult] = pmap(
                type_check_src_in_project,
                [x[0] for x in to_check_values],
                [x[1] for x in to_check_values],
                [x[2] for x in to_check_values],
                [env.pre_fdbks[x[0].file] for x in to_check_values],
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

    file2id = {s.file: i for i, s in enumerate(src_data.all_srcs)}
    src_ids = [file2id[x[0].file] for x in to_check_values]
    critic_inputs_metas = pmap(
        to_critic_inputs,
        [x[0] for x in to_check_values],
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
        collate_fn=CriticCollator(),
    )
    chunk2preds = critic.classify_data(
        dataloader, len(critic_dataset), tqdm_args=tqdm_args
    )
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
        score = np.mean([score_transform(p) for p in all_preds])
        critic_scores.append(float(score))

    scores_map = dict(zip(to_check.keys(), critic_scores))
    preds_map = dict(zip(to_check.keys(), critic_preds))
    final_preds = list[list[PythonType]]()
    extra_info = list[CriticAssesInfo]()
    final_errors = 0
    for i in range(len(chunks.data)):
        candidates = pred_candidates[i]
        cand_scores = [scores_map[(i, j)] for j in range(len(candidates))]
        cand_preds = [preds_map[(i, j)] for j in range(len(candidates))]
        sample_id = int(np.argmax(cand_scores))
        final_errors += n_errors_map[(i, sample_id)]
        final_preds.append(candidates[sample_id])
        extra_info.append(CriticAssesInfo(cand_scores, cand_preds))
    print("Average number of errors after selection:", final_errors / len(chunks.data))
    return DatasetPredResult[CriticAssesInfo](chunks, final_preds, extra_info)


def to_critic_inputs(
    src: TokenizedSrc,
    preds: dict[int, PythonType],
    check_r: SrcCheckResult,
    ctx_args: CtxArgs,
    labels_range: tuple[int, int] | None = None,
):
    """
    Patch each src with the type checker feedbacks and inline the previous predicitons,
    then break the src into one (if short enough) or more chunks.
    """
    errors, current_code = check_r
    fdbks = [] if isinstance(errors, str) else errors
    new_src = feedbacks_to_tokenized_src(
        src, current_code, fdbks, patch_predictions=False
    )
    new_src.prev_types = preds
    new_src = TokenizedSrc.inline_predictions(new_src, as_comment=False)
    chunks = list[dict]()
    chunks_info = list[SrcChunkInfo]()
    if labels_range is None:
        labels_range = min(preds.keys()), max(preds.keys()) + 1
    src_to_chunks_(chunks, chunks_info, new_src, labels_range, ctx_args)
    return chunks, chunks_info


def collect_type_errors_from_predictions(
    src_data: TokenizedSrcSet, result: DatasetPredResult, max_workers: int
) -> list[tuple[Path, MypyFeedback]]:
    "Apply all the predictioins and call the type checker once per project."

    chunks = result.chunks
    chunks_info = chunks.chunks_info
    chunk_preds = result.predictions

    file2src = src_data.file2src(resolve=False)
    srcs_to_check = src_data.all_srcs

    with src_data.setup_typechecking(srcs_to_check) as env:
        to_check = dict[Path, dict[Path, dict[int, str]]]()
        for i in range(len(chunks_info)):
            info = chunks.chunks_info[i]
            file = info.src_file
            src = file2src[file.relative_to(src_data.repos_root)]
            file = src.file
            proj_root = env.template_root / src.repo
            if proj_root not in to_check:
                to_check[proj_root] = dict()
            if file not in to_check[proj_root]:
                to_check[proj_root][file] = dict()
            pred_dict = to_check[proj_root][file]
            for l_id, pred in zip(info.label_ids, chunk_preds[i]):
                pred_dict[l_id] = str(pred)

        check_rs: list[MypyResult | str] = pmap(
            collect_type_errors_in_project,
            [[file2src[f] for f in d.keys()] for d in to_check.values()],
            [[preds for preds in d.values()] for d in to_check.values()],
            [root for root in to_check.keys()],
            max_workers=max_workers,
            desc="map collect_type_errors_in_project",
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
    only_same_file_error: bool = False,
) -> int:
    r = collect_type_errors_in_project(srcs, preds_list, proj_root)
    if isinstance(r, MypyResult):
        if only_same_file_error:
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
) -> MypyResult | str:
    # setup: copy all files into cwd
    proc = multiprocessing.current_process()
    cwd = (project_root.parent.parent / proc.name / project_root.name).resolve()
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
    check_r = MypyChecker.check_project(cwd)
    if isinstance(check_r, MypyResult):
        check_r.error_dict = {
            f.relative_to(cwd): es for f, es in check_r.error_dict.items()
        }
    return check_r
