import copy
import multiprocessing
import pickle
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import *

import dateparser
from datasets import Dataset

from .tokenized_src import *
from .type_check import TypeCheckArgs
from .type_env import (
    AccuracyMetric,
    AnnotCat,
    AnnotInfo,
    AnnotPath,
    MypyChecker,
    MypyFeedback,
    PythonType,
    collect_annots_info,
    normalize_type,
    parse_type_expr,
    parse_type_from_ast,
    type_accuracies,
)
from .utils import *

warnings.filterwarnings(
    "ignore",
    message="The localize method is no longer necessary, as this time zone supports the fold attribute",
)


class TypeCheckSettings:
    temp_path: str = "Default"


@dataclass
class GitRepo:
    author: str
    name: str
    url: str
    stars: int
    forks: int
    description: str = ""
    lines_of_code: Optional[int] = None
    last_update: Optional[datetime] = None
    n_type_annots: Optional[int] = None
    n_type_places: Optional[int] = None

    def authorname(self):
        return self.author + "__" + self.name

    def repo_dir(self, repos_dir: Path) -> Path:
        return repos_dir / "downloaded" / self.authorname()

    def download(self, repos_dir: Path, timeout=None) -> bool:
        subprocess.run(
            ["git", "clone", "--depth", "1", self.url, self.authorname()],
            cwd=(repos_dir / "downloading"),
            timeout=timeout,
            capture_output=True,
        )
        if not (repos_dir / "downloading" / self.authorname()).is_dir():
            # git clone failed. Possibly caused by invalid url?
            return False
        subprocess.run(
            ["mv", self.authorname(), (repos_dir / "downloaded")],
            cwd=(repos_dir / "downloading"),
            capture_output=True,
        )
        return True

    def read_last_update(self, repos_dir):
        d = self.repo_dir(repos_dir)
        s = subprocess.run(
            ["git", "log", "-1", "--format=%cd"], cwd=d, capture_output=True, text=True
        ).stdout
        lu = dateparser.parse(s.split("+")[0])
        assert lu is not None
        self.last_update = lu.replace(tzinfo=None)
        return self.last_update

    def src_files(self, repos_dir):
        for fpath in self.repo_dir(repos_dir).glob("**/*.py"):
            yield (fpath, read_file(fpath))

    def count_lines_of_code(self, repos_dir):
        n_lines = 0
        for src in self.repo_dir(repos_dir).glob("**/*.py"):
            with open(src, "r") as fp:
                n_lines += sum(1 for line in fp if line.rstrip())
        self.lines_of_code = n_lines
        return n_lines

    def collect_annotations(
        self, repos_dir, silent=True
    ) -> dict[Path, dict[AnnotPath, tuple[Optional[PythonType], AnnotCat]]]:
        n_paths, n_annots = 0, 0
        file_to_annots = dict[
            Path, dict[AnnotPath, tuple[Optional[PythonType], AnnotCat]]
        ]()
        for src in self.repo_dir(repos_dir).glob("**/*.py"):
            rpath = src.relative_to(self.repo_dir(repos_dir))
            m = cst.parse_module(read_file(src))
            paths = collect_annots_info(m)
            path_to_cat = {pinfo.path: pinfo.cat for pinfo in paths}
            n_paths += len(paths)
            annots = (info for info in paths if info.annot is not None)
            n_annots += sum(1 for _ in annots)
            file_to_annots[rpath] = {
                (k := info.path): (
                    parse_type_expr(
                        cast(cst.Annotation, info.annot).annotation, silent
                    ),
                    path_to_cat[k],
                )
                for info in annots
            }
        self.n_type_annots = n_annots
        self.n_type_places = n_paths
        return file_to_annots

    def revert_changes(self, repos_dir):
        rd = self.repo_dir(repos_dir)
        result = subprocess.run(
            ["git", "diff", "--name-only"], cwd=rd, capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip() != "":
            print("Reverting changes in", rd)
            subprocess.run(
                ["git", "checkout", "."],
                cwd=rd,
            )

    @staticmethod
    def from_json(json):
        return GitRepo(
            author=json["author"],
            name=json["repo"],
            url=json["repoUrl"],
            stars=json["stars"],
            forks=json["forks"],
        )


@dataclass
class CtxArgs:
    ctx_size: int
    # up to how much of the left_margin to be allocated as preamble
    preamble_size: int
    left_margin: int
    right_margin: int
    max_labels: int = 16
    # whether to inline all the labels that precede the context. Set to true when in interactive mode.
    inline_prev_gold: bool = False

    def __post_init__(self):
        assert self.preamble_size > 0
        assert (
            self.preamble_size < self.left_margin
        ), "Preamble bigger than left_margin.(Preamble is allcoated from the left margin.)"
        assert (
            self.left_margin + self.right_margin < self.ctx_size
        ), "No window size left."

    @property
    def window_size(self) -> int:
        return self.ctx_size - self.left_margin - self.right_margin

    def __repr__(self):
        return repr_modified_args(self)


def _compute_ctx(
    src: TokenizedSrc, label_range: tuple[int, int], ctx_args: CtxArgs
) -> tuple[list[int], tuple[int, int]]:
    src_len = len(src.tokenized_code)
    assert label_range[0] < len(
        src.types_pos
    ), f"label_range={label_range}, len(types_pos)={len(src.types_pos)}"
    window_start = src.types_pos[label_range[0]]
    left_margin_start = max(0, window_start - ctx_args.left_margin)
    left_margin_size = window_start - left_margin_start

    max_window_size = ctx_args.window_size
    right_margin_end = left_margin_start + ctx_args.ctx_size
    if right_margin_end >= src_len:
        right_margin_end = src_len
        max_window_size = right_margin_end - window_start
        assert max_window_size > 0

    label_pos = 0
    label_ids = list[int]()
    assert len(src.types) > 0
    for i in range(label_range[0], label_range[1]):
        label_pos = src.types_pos[i] - window_start
        if 0 <= label_pos < max_window_size:
            label_ids.append(i)
        if len(label_ids) >= ctx_args.max_labels:
            break

    window_size = label_pos + 1
    right_margin_end = min(
        right_margin_end, window_start + window_size + ctx_args.right_margin
    )

    assert right_margin_end - left_margin_start <= ctx_args.ctx_size
    assert left_margin_size <= ctx_args.left_margin
    assert len(label_ids) <= ctx_args.max_labels

    return label_ids, (left_margin_start, right_margin_end)


def chunk_from_src(
    src: TokenizedSrc, label_id: int, ctx_args: CtxArgs
) -> tuple[dict, "SrcChunkInfo"]:
    """Helper function to extract a single chunk from a source file."""
    chunks = list[dict]()
    chunks_info = list[SrcChunkInfo]()
    src_to_chunks_(
        chunks,
        chunks_info,
        src,
        (label_id, label_id + 1),
        ctx_args,
    )
    assert_eq(len(chunks), len(chunks_info), 1)
    return chunks[0], chunks_info[0]


def src_to_chunks(
    src: TokenizedSrc,
    label_range: tuple[int, int],
    ctx_args: CtxArgs,
) -> tuple[list[dict], list["SrcChunkInfo"]]:
    """Helper function to extract chunks from a source file."""
    chunks = list[dict]()
    chunks_info = list[SrcChunkInfo]()
    src_to_chunks_(chunks, chunks_info, src, label_range, ctx_args)
    return chunks, chunks_info


def src_to_chunks_(
    chunks: list[dict],
    chunks_info: list["SrcChunkInfo"],
    src: TokenizedSrc,
    label_range: tuple[int, int],
    ctx_args: CtxArgs,
) -> None:
    assert 0 <= label_range[0]
    assert label_range[1] <= len(
        src.types
    ), f"label_range: {label_range}, len(types): {len(src.types)}"

    tokenizer = DefaultTokenizer
    special_tks = [tokenizer.additional_special_tokens_ids[99 - i] for i in range(100)]
    bos_id, eos_id = not_none(tokenizer.bos_token_id), not_none(tokenizer.eos_token_id)

    if len(src.tokenized_preamble) > ctx_args.preamble_size:
        # cut preamble at the start
        preamble = src.tokenized_preamble[-ctx_args.preamble_size :]
        preamble[0] = bos_id
    else:
        preamble = src.tokenized_preamble
    new_ctx_args = copy.deepcopy(ctx_args)
    new_ctx_args.ctx_size -= len(preamble)
    new_ctx_args.left_margin -= len(preamble)
    new_ctx_args.preamble_size = 0

    label_ids, (ctx_start, ctx_end) = _compute_ctx(src, label_range, new_ctx_args)
    tks = src.tokenized_code[ctx_start:ctx_end]
    if ctx_start > 0:
        tks[0] = bos_id
    if ctx_end < len(src.tokenized_code) - 1:
        tks[-1] = eos_id
    label_tkns = [bos_id]
    types = list[PythonType]()
    types_info = list[AnnotInfo]()
    prev_types = list[PythonType]() if src.prev_types is not None else None
    inlined_spans = list[slice]() if src.inlined_spans is not None else None
    for i, l_id in enumerate(label_ids):
        label_pos = src.types_pos[l_id] - ctx_start
        tks[label_pos] = special_tks[i]
        label_tkns.append(special_tks[i])
        label_tkns.extend(src.types_tks[l_id])
        types.append(src.types[l_id])
        types_info.append(src.types_info[l_id])
        if prev_types is not None and l_id in as_any(src.prev_types):
            prev_types.append(as_any(src.prev_types)[l_id])
        if inlined_spans is not None and l_id in as_any(src.inlined_spans):
            span0 = as_any(src.inlined_spans)[l_id]
            inlined_spans.append(slice(span0.start - ctx_start, span0.stop - ctx_start))
    label_tkns.append(eos_id)

    assert len(label_ids) > 0
    assert len(preamble) + len(tks) <= ctx_args.ctx_size

    this_chunk = {
        "input_ids": preamble + tks,
        "labels": label_tkns,
        "n_labels": len(label_ids),
    }
    chunks.append(this_chunk)

    meta = SrcChunkInfo(
        types,
        types_info,
        src_file=src.file,
        label_ids=label_ids,
        prev_types=prev_types,
        inlined_spans=inlined_spans,
    )
    chunks_info.append(meta)

    if ctx_args.inline_prev_gold:
        prev_types = {t: src.types[t] for t in label_ids}
        src = src.inline_prev_predictions(as_comment=False, prev_types=prev_types)

    new_label_range = (label_ids[-1] + 1, label_range[1])
    if new_label_range[0] < label_range[1]:
        src_to_chunks_(chunks, chunks_info, src, new_label_range, ctx_args)


def chunk_srcs_per_file(
    repos_root: Path,
    srcs: Sequence[TokenizedSrc],
    ctx_args: "CtxArgs",
    tqdm_args: dict,
) -> "ChunkedDataset":
    """Turn each file into a single chunk when possible, or break it down into multiple chunks."""

    # TODO: parallelize this
    srcs = [s for s in srcs if len(s.types) > 0]
    label_ranges = [(0, len(s.types)) for s in srcs]
    chunk_rs = pmap(
        src_to_chunks,
        srcs,
        label_ranges,
        [ctx_args] * len(srcs),
        desc="map src_to_chunks",
        tqdm_args=tqdm_args,
    )

    data: dict[str, list] = {
        "input_ids": [],
        "labels": [],
        "n_labels": [],
        "chunk_id": [],
    }
    chunks = [chunk for r in chunk_rs for chunk in r[0]]
    chunks_info = [info for r in chunk_rs for info in r[1]]

    chunk_id = 0
    for chunk in chunks:
        data["input_ids"].append(chunk["input_ids"])
        data["labels"].append(chunk["labels"])
        data["n_labels"].append(chunk["n_labels"])
        data["chunk_id"].append(chunk_id)
        chunk_id += 1

    files = [(repos_root / s.file).resolve() for s in srcs]
    return ChunkedDataset(
        data=Dataset.from_dict(data),
        chunks_info=chunks_info,
        files=files,
        file2src={f: s.main_code for f, s in zip(files, srcs)},
        file2repo={f: (repos_root / s.repo).resolve() for f, s in zip(files, srcs)},
    )


@dataclass
class TypeCheckingEnv:
    template_root: Path
    pre_fdbks: dict[Path, str | list[MypyFeedback]]


@dataclass
class TokenizedSrcSet:
    repos_root: Path
    all_srcs: list[TokenizedSrc]
    extra_stats: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.all_srcs)

    def inline_predictions(self, as_comment: bool, tqdm_args={}) -> "TokenizedSrcSet":
        new_srcs = pmap(
            TokenizedSrc.inline_predictions,
            self.all_srcs,
            [as_comment] * len(self.all_srcs),
            desc="inline_predictions",
            tqdm_args=tqdm_args,
        )
        return TokenizedSrcSet(
            self.repos_root,
            new_srcs,
            extra_stats=copy.deepcopy(self.extra_stats),
        )

    def common_type_names(self, top_k=100) -> set[str]:
        count = Counter()
        for src in self.all_srcs:
            for t in src.types:
                for n in normalize_type(t).all_names():
                    count[n] += 1
        return {n for n, _ in count.most_common(top_k)}

    def get_src_by_file(self, file: Path) -> TokenizedSrc:
        assert isinstance(file, Path)
        for src in self.all_srcs:
            if src.file == file:
                return src
        raise ValueError(f"No src found for {file}")

    def repos2srcs(self):
        r = groupby(self.all_srcs, lambda s: s.repo)
        for srcs in r.values():
            srcs.sort(key=lambda s: s.file)
        return r

    def add_stats(self, stats: dict, should_print=True):
        if should_print:
            pretty_print_dict(stats)
        self.extra_stats.update(stats)

    def __getitem__(self, ids: slice | Iterable):
        return TokenizedSrcSet(
            self.repos_root,
            get_subset(self.all_srcs, ids),
            {"subset_ids": ids},
        )

    def to_chunks(
        self,
        ctx_args: "CtxArgs",
        tqdm_args: dict = {},
    ) -> "ChunkedDataset":
        srcs = self.all_srcs
        chunks = chunk_srcs_per_file(self.repos_root, srcs, ctx_args, tqdm_args)
        chunks.verify_labels(self, tqdm_args=tqdm_args)
        return chunks

    def file2src(self, resolve=True):
        if resolve:
            return {(self.repos_root / s.file).resolve(): s for s in self.all_srcs}
        else:
            return {s.file: s for s in self.all_srcs}

    def stats_to_show(self) -> dict[str, Any]:
        num_repos = len(set(s.repo for s in self.all_srcs))
        useful_srcs = self.all_srcs
        num_files = len(useful_srcs)
        num_lines = sum(
            len(s.main_code.split("\n")) + len(s.preamble_code.split("\n"))
            for s in useful_srcs
        )
        num_labels = sum(len(s.types) for s in useful_srcs)
        main_tokens_per_file = [len(s.tokenized_code) for s in useful_srcs]
        preamble_tokens_per_file = [len(s.tokenized_preamble) for s in useful_srcs]
        target_tks_per_file = [
            sum(len(tks) + 1 for tks in s.types_tks) for s in useful_srcs
        ]
        basic_stats = {
            "num_repos": num_repos,
            "num_files": num_files,
            "num_lines": num_lines,
            "num_labels": num_labels,
            "main_tokens_per_file": scalar_stats(main_tokens_per_file),
            "preamble_tokens_per_file": scalar_stats(preamble_tokens_per_file),
            "target_tks_per_file": scalar_stats(target_tks_per_file),
        }
        basic_stats.update(self.extra_stats)
        # hide the follwing stats since they are too verbose
        basic_stats.pop("mypy_feedbacks", None)
        basic_stats.pop("check_failure_reasons", None)
        return basic_stats

    def print_stats(self):
        pretty_print_dict(self.stats_to_show())

    @contextmanager
    def setup_typechecking(
        self,
        src_list: Sequence[TokenizedSrc],
        cleanup=True,
        skip_pre_fdbks=False,
    ):
        """
        Context manager to setup the shared files for multi-processing type checking
        and handles the cleanup when done.

        If running multiple instances in parallel, set `TypeCheckSettings.temp_path` to specify which temporary directory (under `mypy_temp`)
        to use.
        """

        # put the temp files out of the project to avoid slowing down VSCode symbol search
        template_root = (
            proj_root()
            / "../mypy_temp"
            / TypeCheckSettings.temp_path
            / f"ORIGINAL_PROJECTS"
        )
        template_root.mkdir(parents=True, exist_ok=True)
        try:
            repo_set = {s.repo for s in src_list}
            repo2srcs = self.repos2srcs()
            for repo in repo_set:
                repo_root = self.repos_root / repo
                # first ensure all files are copied to the template_root
                for f in repo_root.glob("**/*.py"):
                    dest = template_root / repo / f.relative_to(repo_root)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(f, dest)
                # make sure labels are masked out as `Any`s to prevent information leakage
                for s in repo2srcs[repo]:
                    any_preds = {i: "Any" for i, _ in enumerate(s.types)}
                    new_code = code_to_check_from_preds(s, any_preds)
                    new_path = template_root / s.file
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    new_path.write_text(new_code)
            if not skip_pre_fdbks:
                pre_fdbks = self.compute_preexisting_fdbks(src_list, template_root)
            else:
                pre_fdbks = dict()
            yield TypeCheckingEnv(template_root, pre_fdbks)
        finally:
            if cleanup:
                shutil.rmtree(template_root.parent, ignore_errors=True)

    def compute_preexisting_fdbks(
        self, src_list: Sequence[TokenizedSrc], template_root
    ) -> dict[Path, str | list[MypyFeedback]]:
        """Compute the feedbacks caused by predicting all types as `Any`."""

        repo_set = {s.repo for s in src_list}
        repo2id = {repo: i for i, repo in enumerate(repo_set)}
        repo_paths = [template_root / repo for repo in repo_set]
        check_rs = pmap(
            MypyChecker.check_project,
            repo_paths,
            desc="compute_preexisting_fdbks",
        )

        def get_fdbks(src: TokenizedSrc):
            abs_path = (template_root / src.file).resolve()
            check_r = check_rs[repo2id[src.repo]]
            if isinstance(check_r, str):
                return check_r
            else:
                return check_r.error_dict.get(abs_path, [])

        return {s.file: get_fdbks(s) for s in src_list}

    def type_check_each_file_in_project(
        self,
        file2preds: Iterable[tuple[Path, dict[int, str]]],
        tqdm_args={},
    ) -> list["SrcCheckResult"]:
        file2src = self.file2src(resolve=False)
        repos_root = self.repos_root
        src_list = [file2src[f.relative_to(repos_root)] for f, _ in file2preds]

        with self.setup_typechecking(src_list) as env:
            project_roots = [env.template_root / f.repo for f in src_list]

            check_rs: list[SrcCheckResult] = pmap(
                type_check_src_in_project,
                src_list,
                [p for _, p in file2preds],
                project_roots,
                [env.pre_fdbks[s.file] for s in src_list],
                desc="map type_check_src_in_project",
                tqdm_args=tqdm_args,
            )
        return check_rs

    def add_type_checker_feedback(
        self,
        file2preds: dict[Path, dict[int, str]],
        max_workers: int,
        tqdm_args: dict,
        tc_args: TypeCheckArgs,
    ) -> "TokenizedSrcSet":
        """Add the predictions to the corresponding files, call the type checker to
        collect the feedbacks, and then patch the feedbacks as well as the original
        predictions to form the new inputs.
        """

        file2src = self.file2src(resolve=True)
        src_list = [file2src[f.resolve()] for f in file2preds]

        # first, collec type checker feedbacks
        check_rs: list[SrcCheckResult]
        if tc_args.no_feedback:
            check_rs = [
                SrcCheckResult(
                    feedbacks=[], new_code=code_to_check_from_preds(s, preds)
                )
                for s, preds in zip(src_list, list(file2preds.values()))
            ]
        elif tc_args.check_in_isolation:
            check_rs = pmap(
                type_check_src,
                src_list,
                list(file2preds.values()),
                max_workers=max_workers,
                desc="map type_check_src",
                tqdm_args=tqdm_args,
            )
        else:
            check_rs = self.type_check_each_file_in_project(
                file2preds.items(),
                tqdm_args=tqdm_args,
            )

        n_checked = 0
        code_list = list[str]()
        feedback_list = list[list[MypyFeedback]]()
        check_failure_reasons = list[str]()
        for i in range(len(src_list)):
            errors, new_code = check_rs[i]
            if isinstance(errors, str):
                check_failure_reasons.append(errors)
                errors = []
            else:
                n_checked += 1
            code_list.append(new_code)
            feedback_list.append(errors)
        result = TokenizedSrcSet(self.repos_root, [], copy.deepcopy(self.extra_stats))
        silent = tqdm_args.get("disable", False)
        result.add_stats(
            {
                "type_check_success_ratio": n_checked / len(src_list),
                "feedbacks_per_file": scalar_stats([len(fs) for fs in feedback_list]),
            },
            not silent,
        )
        result.add_stats(
            {
                "check_failure_reasons": check_failure_reasons,
                "mypy_feedbacks": feedback_list,
            },
            should_print=False,
        )

        # then, patch the srcs with the feedbacks and predictions to form new srcs
        new_srcs = pmap(
            feedbacks_to_tokenized_src,
            src_list,
            code_list,
            feedback_list,
            max_workers=max_workers,
            desc="feedbacks_to_tokenized_src",
            tqdm_args=tqdm_args,
        )
        result.all_srcs = new_srcs
        # assert_eq(len(new_srcs), len(file2preds))
        return result

    def __repr__(self):
        return f"TokenizedSrcSet(root='{self.repos_root}', n_repos={len(self.repos2srcs())}, n_labeled_files={len(self.all_srcs)})"

    @staticmethod
    def from_repos(
        repos_root: Path,
        repos_paths: Iterable[Path],
        preprocess_args: PreprocessArgs,
        max_workers: int | None = None,
        tqdm_args: dict = {},
        max_line_width: int = 400,
        ignore_dirs: set[str] = {".venv", ".mypy_cache", ".git", "venv"},
    ) -> "TokenizedSrcSet":
        for r in repos_paths:
            assert r.is_dir(), f"Provided path {r} is not a directory."

        # file_path, (code, repo_path)
        srcs: dict[Path, tuple[str, Path]] = {
            f: (f.read_text(), r)
            for r in repos_paths
            for f in rec_iter_files(r, dir_filter=lambda d: d.name not in ignore_dirs)
            if f.suffix == ".py" and not f.is_symlink()
        }
        num_all_srcs = len(srcs)

        def file_width(text):
            return max(len(l) for l in text.split("\n"))

        srcs = {
            f: (code, r)
            for f, (code, r) in srcs.items()
            if file_width(code) <= max_line_width
        }
        result = TokenizedSrcSet(repos_root, [])
        result.add_stats(
            {
                "n_files_too_wide": num_all_srcs - len(srcs),
                "too_wide_ratio": (1 - len(srcs) / num_all_srcs),
                "preprocess": preprocess_args,
            }
        )

        def avoid_type_mask(code: str):
            return code.replace(SpecialNames.TypeMask, "MaskReplaced")

        code_list = [avoid_type_mask(x[0]) for x in srcs.values()]
        file_list = list(srcs.keys())
        repo_list = [x[1] for x in srcs.values()]
        parsing_results = pmap(
            _try_parse_src,
            code_list,
            file_list,
            repo_list,
            [preprocess_args] * len(srcs),
            max_workers=max_workers,
            desc="parse src code",
            tqdm_args=tqdm_args,
        )

        filtered_srcs = []
        for x in parsing_results:
            if x is None or len(x.types) == 0:
                continue
            x.file = x.file.relative_to(repos_root)
            x.repo = x.repo.relative_to(repos_root)
            filtered_srcs.append(x)

        for f, g in groupby(filtered_srcs, lambda s: s.file).items():
            assert len(g) == 1, f"{f} appears {len(g)} times."

        result.all_srcs = filtered_srcs
        return result


def _try_parse_src(
    code: str, file: Path, repo: Path, args: PreprocessArgs
) -> Optional[TokenizedSrc]:
    try:
        return TokenizedSrc.parse(code, file, repo, args)
    except cst.ParserSyntaxError as e:
        return None


def create_tokenized_srcsets(
    dataset: str,
    out_dir: Path,
    func_only: bool,
    pre_args: PreprocessArgs,
    data_reduction: int = 1,
) -> None:
    import typet5.function_dataset as fd

    repos_dir = get_dataset_dir(dataset) / "repos"
    out_dir.mkdir(parents=True, exist_ok=True)

    tk_dataset: dict[str, TokenizedSrcSet] = {}
    with run_long_task(f"Generating TokenizedSrcSets: {out_dir.name}", notify=False):
        for name in ["test", "train", "valid"]:
            base = repos_dir / name
            if not base.exists():
                print(f"[Warning] Split {name} not found. Skip.")
                continue

            print(f"Creating: {name}")
            repo_roots = [d for d in base.iterdir() if d.is_dir()]
            if name == "train":
                n_train = len(repo_roots) // data_reduction
                repo_roots = repo_roots[:n_train]
            if func_only:
                tk_data = fd.dataset_from_repos(base, repo_roots, pre_args)
            else:
                tk_data = TokenizedSrcSet.from_repos(base, repo_roots, pre_args)
            for s in tk_data.all_srcs:
                assert len(s.types) > 0, f"{s.file} has no labels."
            tk_dataset[name] = tk_data
            save_path = out_dir / f"{name}.pkl"
            pickle_dump(save_path, tk_data)
            stats_str = pretty_show_dict(tk_data.stats_to_show())
            write_file(out_dir / f"{name}-stats.txt", stats_str)
            print(f"Saved to {save_path}")
        assert tk_dataset, "Empty dataset."


def load_tokenized_srcsets(
    path: Path,
    quicktest: bool = False,
    sets_to_load=["test", "train", "valid"],
) -> dict[str, TokenizedSrcSet]:
    print("Loading TokenizedSrcSets: ", path)
    subprocess.run(["du", "-sh", path])
    tk_dataset = dict[str, TokenizedSrcSet]()
    for n in sets_to_load:
        file = path / f"{n}.pkl"
        if not file.exists():
            print(f"{file} not found. Skip.")
            continue
        sdata = pickle_load(file)
        if quicktest:
            ids = range(0, len(sdata.all_srcs), max(1, len(sdata.all_srcs) // 10))
            sdata = sdata[ids]
        tk_dataset[n] = sdata
    assert tk_dataset, "Empty dataset."
    return tk_dataset


def code_to_check_from_preds(
    src: TokenizedSrc, preds: dict[int, str] | dict[int, PythonType]
):
    code = src.main_code
    if not preds:
        return code
    changes = list[tuple[CodeRange, int, str]]()
    start = CodePosition(0, 0)
    changes.append((CodeRange(start, start), 0, MypyChecker.Preamble))
    for k in preds.keys():
        assert k in range(len(src.types)), f"Prediction index out of range: {k}"
    for i, info in enumerate(src.types_info):
        r = not_none(info.annot_range)
        pred = str(preds.get(i, "Any"))
        changes.append((r, 1, pred))
    new_code = replace_strs_by_pos(code, changes)
    return new_code


class SrcCheckResult(NamedTuple):
    feedbacks: list[MypyFeedback] | str
    new_code: str

    def pretty_print(self):
        print("Feedbacks:")
        for f in self.feedbacks:
            print(f)
        print("======= New code =======")
        print(add_line_numbers(self.new_code))

    def remove_preexisting(self, pre: list[MypyFeedback] | str) -> "SrcCheckResult":
        if isinstance(pre, list) and isinstance(self.feedbacks, list):
            preexisting = {(f.position.line, f.message) for f in pre}
            new_fdbk = [
                f
                for f in self.feedbacks
                if (f.position.line, f.message) not in preexisting
            ]
            return SrcCheckResult(new_fdbk, self.new_code)
        else:
            return self


def type_check_src_skip_check(
    src: TokenizedSrc,
    preds: dict[int, str],
) -> SrcCheckResult:
    new_code = code_to_check_from_preds(src, preds)
    return SrcCheckResult([], new_code)


def type_check_src(
    src: TokenizedSrc,
    preds: dict[int, str],
    cwd: Optional[Path] = None,
) -> SrcCheckResult:
    def from_preds(preds: dict[int, str]):
        new_code = code_to_check_from_preds(src, preds)
        check_r = MypyChecker.check_code(new_code, cwd=cwd)
        feedback: list[MypyFeedback] | str
        if isinstance(check_r, str):
            feedback = check_r
        elif len(check_r.error_dict) == 0:
            feedback = []
        else:
            assert len(check_r.error_dict) == 1
            feedback = list(check_r.error_dict.values())[0]
        return feedback, new_code

    fdbk0, _ = from_preds({i: "Any" for i, _ in preds.items()})
    fdbk1, new_code = from_preds(preds)
    if isinstance(fdbk1, list) and isinstance(fdbk0, list):
        preexisting = {(f.position.line, f.message) for f in fdbk0}
        new_fdbk = [f for f in fdbk1 if (f.position.line, f.message) not in preexisting]
        return SrcCheckResult(new_fdbk, new_code)
    else:
        return SrcCheckResult(fdbk1, new_code)


def type_check_src_in_project(
    src: TokenizedSrc,
    preds: dict[int, str] | dict[int, PythonType],
    project_root: Path,
    preexisting: list[MypyFeedback] | str,
) -> SrcCheckResult:
    # setup: copy all files into cwd
    proc = multiprocessing.current_process()
    cwd = project_root.parent.parent / proc.name / project_root.name
    cwd.mkdir(parents=True, exist_ok=True)

    for f in project_root.glob("**/*.py"):
        rel_path = f.relative_to(project_root)
        (cwd / rel_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, cwd / rel_path)

    rel_path = src.file.relative_to(src.repo)
    file_path = cwd / rel_path

    def from_preds(preds: dict[int, str] | dict[int, PythonType]):
        new_code = code_to_check_from_preds(src, preds)
        file_path.write_text(new_code)
        check_r = MypyChecker.check_project(cwd)
        feedback: list[MypyFeedback] | str
        if isinstance(check_r, str):
            feedback = check_r
        else:
            feedback = check_r.error_dict.get(file_path.resolve(), [])
        return SrcCheckResult(feedback, new_code)

    return from_preds(preds).remove_preexisting(preexisting)


@dataclass
class SrcChunkInfo:
    """Stores the source code information for a chunk of tokens."""

    types: list[PythonType]  # the label types in this chunk
    annots_info: list[AnnotInfo]  # the label AnnotInfos in this chunk
    src_file: Path
    # maps each label to its label id in the corresponding TokenizedSrc
    label_ids: list[int]
    prev_types: list[PythonType] | None
    inlined_spans: list[slice] | None

    def __repr__(self):
        return f"SrcChunkInfo(num_types={len(self.types)}, src_file='{self.src_file}')"


@dataclass
class ChunkedDataset:
    data: Dataset
    chunks_info: list[SrcChunkInfo]
    # The source files of this data set
    files: list[Path]
    file2src: dict[Path, str]
    file2repo: dict[Path, Path]

    def __post_init__(self):
        assert_eq(len(self.data), len(self.chunks_info))

    def __getitem__(self, chunk_ids: Iterable[int]) -> "ChunkedDataset":
        cid2id = {bid: i for i, bid in enumerate(self.data["chunk_id"])}
        ids = [cid2id[bid] for bid in chunk_ids]

        new_data = {n: get_subset(self.data[n], ids) for n in self.data.column_names}
        new_info = get_subset(self.chunks_info, ids)

        return ChunkedDataset(
            Dataset.from_dict(new_data),
            chunks_info=new_info,
            files=self.files,
            file2src=self.file2src,
            file2repo=self.file2repo,
        )

    def __len__(self):
        assert_eq(len(self.data), len(self.chunks_info))
        return len(self.data)

    def __repr__(self):
        return f"ChunkedDataset(num_chunks={len(self.chunks_info)}, num_srcs={len(self.files)})"

    def verify_labels(self, srcs: TokenizedSrcSet, tqdm_args={}):
        """
        Verify that the labels in the dataset match the source code.
        """

        src_path_map = dict[Path, dict[AnnotPath, PythonType]]()
        for f, src in tqdm(
            srcs.file2src(resolve=False).items(), desc="building label map", **tqdm_args
        ):
            src_path_map[f] = {
                info.path: ty for ty, info in zip(src.types, src.types_info)
            }
            assert_eq(len(src_path_map[f]), len(src.types))
        input_ids = tqdm(self.data["input_ids"], desc="verify_labels", **tqdm_args)
        for input, chunk in zip(input_ids, self.chunks_info):
            file = chunk.src_file
            for info, ty in zip(chunk.annots_info, chunk.types):
                assert file in src_path_map, f"{file} not in file2src."
                assert (
                    info.path in src_path_map[file]
                ), f"{info.path} should not be a label in {file}.\nExpected label map: {src_path_map[file]}\nChunk code:\n{decode_tokens(input)}"
                assert_eq(
                    src_path_map[file][info.path],
                    ty,
                    extra_message=lambda: f"file={file}, path={info.path}",
                )


def output_ids_as_seqs(output_ids: Iterable[int]):
    """Divide the model output as a sequence of tokens, filtering out padding tokens."""
    seq_id = 0
    buff = list[int]()
    seqs = list[list[int]]()
    tokenizer = DefaultTokenizer
    mark = tokenizer.additional_special_tokens_ids[99 - seq_id]

    for tk in output_ids:
        if tk <= 0:
            continue  # pad or masked token
        if tk != mark:
            buff.append(tk)
        else:
            seqs.append(buff)
            buff = []
            seq_id += 1
            mark = tokenizer.additional_special_tokens_ids[99 - seq_id]
    seqs.append(buff)
    return seqs[1:]


def output_ids_as_types(output_ids: Iterable[int], n_types: int) -> list[PythonType]:
    """Try to parse model outputs as a list of Python types, pad `Any` to make sure the
    list is of the correct length."""
    seqs = output_ids_as_seqs(output_ids)
    tokenizer = DefaultTokenizer
    types = list[PythonType]()
    for seq in seqs[:n_types]:
        try:
            ex_str = tokenizer.decode(seq, skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Failed to decode sequence: {seq}") from e
        try:
            tree = ast.parse(ex_str, mode="eval").body
            ty = parse_type_from_ast(tree)
        except:
            ty = PythonType.Any()
        assert (
            ty.__class__.__name__ == PythonType.__name__
        ), f"{ty} of type {type(ty)} is not a PythonType."
        types.append(ty)
    types.extend(PythonType.Any() for _ in range(n_types - len(types)))
    assert len(types) == n_types
    return types


def R1_srcs_from_preds(
    r0_src: TokenizedSrcSet,
    chunks_info: list[SrcChunkInfo],
    src_files: list[Path],
    r0_preds: list[list[PythonType]],
    tc_args: TypeCheckArgs,
    max_workers: int,
    tqdm_args: dict = {},
) -> TokenizedSrcSet:
    file2preds = dict[Path, dict[AnnotPath, str]]()
    for preds, chunk_info in zip(r0_preds, chunks_info):
        assert_eq(len(preds), len(chunk_info.types))
        file = (r0_src.repos_root / chunk_info.src_file).resolve()
        for i, pred in enumerate(preds):
            if file not in file2preds:
                file2preds[file] = dict()
            label_path = chunk_info.annots_info[i].path
            file2preds[file][label_path] = str(pred)

    file2src = r0_src.file2src()
    file2preds1 = dict[Path, dict[int, str]]()

    for f, ls in file2preds.items():
        src = file2src[f]
        path2id = {info.path: i for i, info in enumerate(src.types_info)}
        try:
            file2preds1[f] = {path2id[path]: label for path, label in ls.items()}
        except Exception as e:
            raise RuntimeError(f"In file {f}. path2id={path2id}") from e

    # assert_eq(len(file2preds1), len(r0_src.all_srcs))
    return r0_src.add_type_checker_feedback(
        file2preds1,
        tc_args=tc_args,
        max_workers=max_workers,
        tqdm_args=tqdm_args,
    )


def preds_to_accuracies(
    preds: Sequence[Sequence[PythonType]],
    dataset: ChunkedDataset,
    metric: AccuracyMetric,
):
    cats = [an.cat for info in dataset.chunks_info for an in info.annots_info]
    labels = [ty for info in dataset.chunks_info for ty in info.types]
    # poses = [i for info in dataset.chunks_info for i in info.label_ids]
    return type_accuracies(
        list(seq_flatten(preds)),
        labels,
        cats,
        metric=metric,
    )


def src_preds_to_accuracies(
    preds: Sequence[dict[int, PythonType]] | Sequence[Sequence[PythonType]],
    srcs: Sequence[TokenizedSrc],
    metric: AccuracyMetric,
):
    pred_types = list[PythonType]()
    cats, labels, poses = [], [], []

    for pred, src in zip(preds, srcs):
        if not isinstance(pred, dict):
            pred = {i: p for i, p in enumerate(pred)}
        for t, ty in pred.items():
            pred_types.append(ty)
            cats.append(src.types_info[t].cat)
            labels.append(src.types[t])
            poses.append(t)

    return type_accuracies(
        pred_types,
        labels,
        cats,
        metric=metric,
    )


def get_tk_dataset_name(
    dataset: str,
    pre_args: PreprocessArgs,
    func_only: bool,
    data_reduction: int = 1,
):
    reduction_tag = f"-reduction={data_reduction}" if data_reduction != 1 else ""
    pre_parts = repr_modified_args(pre_args)
    if func_only:
        return f"func-{dataset}-v7{reduction_tag}-{pre_parts}"
    else:
        return f"{dataset}-v5{reduction_tag}-{pre_parts}"
