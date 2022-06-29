import logging
import math
import multiprocessing
import pickle
import random
import shutil
import subprocess
import warnings
import copy
from dataclasses import dataclass
from datetime import datetime
from typing import *

import dateparser
from datasets import Dataset

from spot.type_check import MypyResult, TypeCheckArgs, parse_type_str
from spot.type_env import (
    AnnotCat,
    AnnotInfo,
    AnnotPath,
    MypyChecker,
    MypyFeedback,
    PythonType,
    apply_annotations,
    collect_annots_info,
    collect_user_annotations,
    normalize_type,
    parse_type_expr,
    parse_type_from_ast,
)
from spot.utils import *

warnings.filterwarnings(
    "ignore",
    message="The localize method is no longer necessary, as this time zone supports the fold attribute",
)


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
                        m, cast(cst.Annotation, info.annot).annotation, silent
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
class TokenizedSrc:
    """A src file with certain type annotations masked out."""

    file: Path
    repo: Path
    types: list[PythonType]
    types_pos: list[int]  # the position of the types in tokenized_code.
    types_str: list[str]
    types_tks: list[list[int]]
    types_info: list[AnnotInfo]
    origin_code: str
    tokenized_code: list[int]  # with certain types masked out
    prev_types: dict[int, PythonType] | None = None  # previously predicted types
    inlined_spans: dict[int, slice] | None = None  # the spans of inlined previous types

    def inline_prev_predictions(self, as_comment: bool) -> "TokenizedSrc":
        "Inine the previous predictions into the code, either directly or as comments."
        prev_types = self.prev_types
        assert isinstance(prev_types, dict), f"prev_types has type: {type(prev_types)}"
        assert len(prev_types) > 0

        types_pos = list[int]()
        tokenized_code = list[int]()
        tokenizer = load_tokenizer_spot()
        mask_id = tokenizer.mask_token_id
        comment_start = tokenizer.encode("/* ", add_special_tokens=False)
        comment_end = tokenizer.encode(" */", add_special_tokens=False)

        inlined_spans = dict[int, slice]()

        i_types = 0
        for tk in self.tokenized_code:
            tokenized_code.append(tk)
            if tk == mask_id:
                span_start = len(tokenized_code)
                types_pos.append(span_start - 1)

                if i_types in prev_types:
                    to_insert = tokenizer.encode(
                        str(prev_types[i_types]), add_special_tokens=False
                    )
                    if as_comment:
                        to_insert = comment_start + to_insert + comment_end
                    tokenized_code.extend(to_insert)
                    span_end = len(tokenized_code)
                    inlined_spans[i_types] = slice(span_start, span_end)
                    assert_eq(tokenized_code[span_start:span_end], to_insert)
                i_types += 1
        assert_eq(i_types, len(self.types))
        assert prev_types.keys() == inlined_spans.keys()

        return TokenizedSrc(
            file=self.file,
            repo=self.repo,
            types=self.types,
            types_pos=types_pos,
            types_str=self.types_str,
            types_tks=self.types_tks,
            types_info=self.types_info,
            origin_code=self.origin_code,
            tokenized_code=tokenized_code,
            prev_types=prev_types,
            inlined_spans=inlined_spans,
        )

    def print_code(self):
        "Print out the (decoded) token sequence"
        return print(decode_tokens(self.tokenized_code))

    @staticmethod
    def inline_predictions(src: "TokenizedSrc", as_comment=False):
        return src.inline_prev_predictions(as_comment=as_comment)


@dataclass
class CtxArgs:
    ctx_size: int
    left_margin: int
    right_margin: int
    max_labels: int = 16

    @property
    def window_size(self) -> int:
        return self.ctx_size - self.left_margin - self.right_margin

    def as_tuple(self):
        "Returns (left_margin, window_size, right_margin, max_labels)."
        return self.left_margin, self.window_size, self.right_margin, self.max_labels

    def __repr__(self):
        return repr_modified_args(self)


class _TokenizedSrcHelper:
    tokenizer: TokenizerSPOT

    def __init__(self, tokenizer: TokenizerSPOT):
        _turn_off_tokenizer_warning(tokenizer)
        self.tokenizer = tokenizer

    def dict_to_tokenized_src(self, d: dict) -> TokenizedSrc:
        r = TokenizedSrc(
            file=d["file"],
            repo=d["repo"],
            origin_code=d["cst_code"],
            tokenized_code=list[int](),
            types=list[PythonType](),
            types_pos=list[int](),
            types_str=list[str](),
            types_info=list[AnnotInfo](),
            types_tks=list[list[int]](),
            prev_types=d["prev_types"],
        )

        match d:
            case {
                "code_segs": segs,
                "types": types,
                "types_str": types_str,
                "annots_info": annots_info,
                "is_label": is_label,
            }:
                assert len(segs) == len(types) + 1
            case _:
                raise ValueError(f"Invalid dict with keys: {d.keys()}")

        tkn = self.tokenizer
        bos_id = not_none(tkn.bos_token_id)
        eos_id = not_none(tkn.eos_token_id)
        mask_id = not_none(tkn.mask_token_id)
        all_tks = r.tokenized_code
        all_tks.append(bos_id)
        for i in range(len(types)):
            all_tks.extend(tkn.encode(segs[i], add_special_tokens=False))
            if is_label is None or is_label[i]:
                r.types_pos.append(len(all_tks))
                r.types.append(types[i])
                r.types_tks.append(tkn.encode(str(types[i]), add_special_tokens=False))
                r.types_str.append(types_str[i])
                r.types_info.append(annots_info[i])
                all_tks.append(mask_id)
            else:
                all_tks.extend(tkn.encode(types_str[i], add_special_tokens=False))
        all_tks.extend(tkn.encode(segs[-1], add_special_tokens=False))
        all_tks.append(eos_id)

        return r

    def feedbacks_to_tokenized_src(
        self,
        src: TokenizedSrc,
        current_code: str,
        feedbacks: list[MypyFeedback],
        patch_predictions: bool = False,
    ) -> TokenizedSrc:
        try:
            m = cst.parse_module(current_code)
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse file: '{src.file}' with content:\n{current_code}"
            ) from e
        m_code = m.code
        assert (
            m_code.rstrip() == current_code.rstrip()
        ), f"String diffferences: {show_string_diff(current_code, m_code)}"
        current_annots, _ = collect_user_annotations(m)
        preds_map = dict[CodeRange, str]()
        types = list[PythonType]()
        prev_types = dict[int, PythonType]()
        types_str = list[str]()
        annots_info = list[AnnotInfo]()
        path2label_id = {info.path: i for i, info in enumerate(src.types_info)}

        for a in current_annots:
            if a.path in path2label_id:
                assert (r := a.annot_range) is not None
                assert (annot := a.annot) is not None
                prev_type = preds_map[r] = m.code_for_node(annot.annotation)
                li = path2label_id[a.path]
                prev_types[li] = parse_type_str(prev_type)
                types.append(src.types[li])
                types_str.append(src.types_str[li])
                annots_info.append(a)
        pos_to_msg = {f.position: f.message for f in feedbacks}
        new_code = patch_code_with_extra(
            current_code, preds_map, pos_to_msg, patch_predictions
        )
        code_segs = new_code.split(SpecialNames.TypeMask)
        assert len(code_segs) == len(types) + 1, f"{len(code_segs)} != {len(types)} + 1"

        d = {
            "file": src.file,
            "repo": src.repo,
            "cst_code": new_code,
            "code_segs": code_segs,
            "types": types,
            "types_str": types_str,
            "annots_info": annots_info,
            "prev_types": prev_types,
            "is_label": None,
        }
        return self.dict_to_tokenized_src(d)


def _compute_ctx(
    src: TokenizedSrc, label_range: tuple[int, int], ctx_args: "CtxArgs"
) -> tuple[list[int], tuple[int, int]]:
    src_len = len(src.tokenized_code)
    assert label_range[0] < len(
        src.types_pos
    ), f"label_range={label_range}, len(types_pos)={len(src.types_pos)}"
    window_start = src.types_pos[label_range[0]]
    left_margin_start = max(0, window_start - ctx_args.left_margin)
    left_margin_size = window_start - left_margin_start

    max_window_size = ctx_args.window_size
    label_ids = list[int]()

    right_margin_end = left_margin_start + ctx_args.ctx_size
    if right_margin_end >= src_len:
        right_margin_end = src_len
        max_window_size = right_margin_end - window_start
        assert max_window_size > 0

    label_pos = 0
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


def src_to_chunks_(
    chunks: list,
    chunks_info: list,
    src: TokenizedSrc,
    src_id: int,
    label_range: tuple[int, int],
    ctx_args: "CtxArgs",
    tokenizer: TokenizerSPOT,
) -> None:
    assert 0 <= label_range[0]
    assert label_range[1] <= len(
        src.types
    ), f"label_range: {label_range}, len(types): {len(src.types)}"

    special_tks = [tokenizer.additional_special_tokens_ids[99 - i] for i in range(100)]
    bos_id, eos_id = not_none(tokenizer.bos_token_id), not_none(tokenizer.eos_token_id)

    label_ids, (ctx_start, ctx_end) = _compute_ctx(src, label_range, ctx_args)
    tks = src.tokenized_code[ctx_start:ctx_end]
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
        if prev_types is not None:
            prev_types.append(not_none(src.prev_types)[l_id])
        if inlined_spans is not None:
            span0 = not_none(src.inlined_spans)[l_id]
            inlined_spans.append(slice(span0.start - ctx_start, span0.stop - ctx_start))
    label_tkns.append(eos_id)

    assert len(label_ids) > 0
    assert len(tks) <= ctx_args.ctx_size
    this_chunk = {
        "input_ids": tks,
        "labels": label_tkns,
        "n_labels": len(label_ids),
    }
    chunks.append(this_chunk)

    meta = SrcChunkInfo(
        types,
        types_info,
        src_ids=[src_id] * len(label_ids),
        label_ids=label_ids,
        prev_types=prev_types,
        inlined_spans=inlined_spans,
    )
    chunks_info.append(meta)

    new_label_range = (label_ids[-1] + 1, label_range[1])
    if new_label_range[0] < label_range[1]:
        src_to_chunks_(
            chunks, chunks_info, src, src_id, new_label_range, ctx_args, tokenizer
        )


def chunk_srcs_per_file(
    repos_root: Path,
    srcs: Sequence[TokenizedSrc],
    ctx_args: "CtxArgs",
    tokenizer: TokenizerSPOT,
    tqdm_args: dict,
) -> "ChunkedDataset":
    """Turn each file into a single chunk when possible, or break it down into multiple chunks."""

    chunks = list[dict]()
    chunks_info: list[SrcChunkInfo] = []

    for src_id, src in enumerate(tqdm(srcs, desc="chunk_srcs_per_file", **tqdm_args)):
        if len(src.types) == 0:
            continue
        labels_range = 0, len(src.types)
        src_to_chunks_(
            chunks, chunks_info, src, src_id, labels_range, ctx_args, tokenizer
        )

    data: dict[str, list] = {
        "input_ids": [],
        "labels": [],
        "n_labels": [],
        "chunk_id": [],
    }
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
        file2src={f: s.origin_code for f, s in zip(files, srcs)},
        file2repo={f: (repos_root / s.repo).resolve() for f, s in zip(files, srcs)},
        tokenizer=tokenizer,
    )


@dataclass
class SrcDataset:
    repos_root: Path
    all_srcs: list[TokenizedSrc] = field(default_factory=list)
    extra_stats: dict = field(default_factory=dict)
    predictions_inlined: bool = False

    def inline_predictions(self, tqdm_args={}) -> "SrcDataset":
        assert not self.predictions_inlined
        new_srcs = pmap(
            TokenizedSrc.inline_predictions,
            self.all_srcs,
            desc="inline_predictions",
            tqdm_args=tqdm_args,
        )
        return SrcDataset(
            repos_root=self.repos_root,
            all_srcs=new_srcs,
            extra_stats=copy.deepcopy(self.extra_stats),
            predictions_inlined=True,
        )

    def file2id(self) -> dict[Path, int]:
        return {s.file: i for i, s in enumerate(self.all_srcs)}

    def id2src(self) -> dict[int, TokenizedSrc]:
        return {i: s for i, s in enumerate(self.all_srcs)}

    def get_src_by_file(self, rel_path: Path) -> TokenizedSrc:
        assert isinstance(rel_path, Path)
        for src in self.all_srcs:
            if src.file.relative_to(src.repo) == rel_path:
                return src
        raise ValueError(f"No src found for {rel_path}")

    def repos2srcs(self):
        r = groupby(self.all_srcs, lambda s: s.repo)
        for srcs in r.values():
            srcs.sort(key=lambda s: s.file)
        return r

    def srcs_with_labels(self):
        "Returns all srcs with at least one label type in it."
        return [s for s in self.all_srcs if len(s.types) > 0]

    def add_stats(self, stats: dict, should_print=True):
        if should_print:
            pretty_print_dict(stats)
        self.extra_stats.update(stats)

    def __getitem__(self, ids: slice | Iterable):
        return SrcDataset(
            self.repos_root, get_subset(self.all_srcs, ids), {"subset_ids": ids}
        )

    def to_chunks(
        self,
        tokenizer: TokenizerSPOT,
        ctx_args: "CtxArgs",
        tqdm_args: dict = {},
    ) -> "ChunkedDataset":
        srcs = self.srcs_with_labels()
        chunks = chunk_srcs_per_file(
            self.repos_root, srcs, ctx_args, tokenizer, tqdm_args
        )
        chunks.verify_labels(self, tokenizer, tqdm_args=tqdm_args)
        return chunks

    def file2src(self, resolve=True):
        if resolve:
            return {(self.repos_root / s.file).resolve(): s for s in self.all_srcs}
        else:
            return {s.file: s for s in self.all_srcs}

    def stats_to_show(self) -> dict[str, Any]:
        num_repos = len(set(s.repo for s in self.all_srcs))
        useful_srcs = self.srcs_with_labels()
        num_files = len(useful_srcs)
        num_lines = sum(len(s.origin_code.split("\n")) for s in useful_srcs)
        num_labels = sum(len(s.types) for s in useful_srcs)
        tokens_per_file = [len(s.tokenized_code) for s in useful_srcs]
        target_tks_per_file = [
            sum(len(tks) + 1 for tks in s.types_tks) for s in useful_srcs
        ]
        basic_stats = {
            "num_repos": num_repos,
            "num_files": num_files,
            "num_lines": num_lines,
            "num_labels": num_labels,
            "tokens_per_file": scalar_stats(tokens_per_file),
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
    def prepare_typecheck_projects(self, src_list: Sequence[TokenizedSrc]):
        # prepare project files
        template_root = MypyChecker.temp_dir() / "original_projects"
        template_root.mkdir(parents=True, exist_ok=True)
        try:

            repo_set = {s.repo for s in src_list}
            repo2srcs = self.repos2srcs()
            for repo in repo_set:
                for s in repo2srcs[repo]:
                    any_preds = {i: "Any" for i, _ in enumerate(s.types)}
                    new_code = code_to_check_from_preds(s, any_preds)
                    new_path = template_root / s.file
                    new_path.parent.mkdir(parents=True, exist_ok=True)
                    new_path.write_text(new_code)
            yield template_root
        finally:
            shutil.rmtree(MypyChecker.temp_dir(), ignore_errors=True)

    def type_check_each_file_in_project(
        self,
        file2preds: Iterable[tuple[Path, dict[int, str]]],
        mypy_path=None,
        include_all_errors: bool = False,
        tqdm_args={},
    ) -> list["SrcCheckResult"]:
        file2src = self.file2src(resolve=False)
        repos_root = self.repos_root
        src_list = [file2src[f.relative_to(repos_root)] for f, _ in file2preds]

        with self.prepare_typecheck_projects(src_list) as template_root:
            project_roots = [template_root / f.repo for f in src_list]

            check_rs: list[SrcCheckResult] = pmap(
                type_check_src_in_project,
                src_list,
                [p for _, p in file2preds],
                project_roots,
                [include_all_errors for _ in src_list],
                [mypy_path for _ in src_list],
                desc="map type_check_src_in_project",
                tqdm_args=tqdm_args,
            )
        return check_rs

    def add_type_checker_feedback(
        self,
        tokenizer: TokenizerSPOT,
        file2preds: dict[Path, dict[int, str]],
        max_workers: int,
        tqdm_args: dict,
        tc_args: TypeCheckArgs,
        include_all_errors: bool = False,
        mypy_path: Optional[Path] = None,
    ) -> "SrcDataset":
        """Add the predictions to the corresponding files, call the type checker to
        collect the feedbacks, and then patch the feedbacks as well as the original
        predictions to form the new inputs.

        If in_isolation is True, then each file is treated as a single-file project.
        This can lead to better performance but is less precise.
        """

        file2src = self.file2src()
        src_list = [file2src[f.resolve()] for f in file2preds]

        # first, collec type checker feedbacks
        try:
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
                    [mypy_path for _ in src_list],
                    max_workers=max_workers,
                    desc="map type_check_src",
                    tqdm_args=tqdm_args,
                )
            else:
                check_rs = self.type_check_each_file_in_project(
                    file2preds.items(),
                    mypy_path,
                    include_all_errors=include_all_errors,
                    tqdm_args=tqdm_args,
                )
        finally:
            MypyChecker.clear_temp_cache()
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
        result = SrcDataset(self.repos_root)
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
        helper = _TokenizedSrcHelper(tokenizer)
        new_srcs = pmap(
            helper.feedbacks_to_tokenized_src,
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
        return f"SrcDataset(root='{self.repos_root}', n_repos={len(self.repos2srcs())}, n_labeled_files={len(self.srcs_with_labels())})"

    @staticmethod
    def from_repos(
        repos_root: Path,
        repos_paths: Iterable[Path],
        tokenizer: TokenizerSPOT,
        drop_comments: bool,
        max_workers: int,
        label_ratio: float = 0.5,
        tqdm_args: dict = {},
        max_line_width: int = 200,
        seed: int = 42,
    ) -> "SrcDataset":
        """Generate the dataset by randomly mask out a fraction of the type annotations as labels.
        If keep_comments if False, will also remove all comments and docstrings.
        """

        # file_path, code, repo_path
        srcs: dict[Path, tuple[str, Path]] = {
            f: (f.read_text(), r)
            for r in repos_paths
            for f in sorted(r.glob("**/*.py"))
            if not f.is_symlink()
        }
        num_all_srcs = len(srcs)

        def file_width(text):
            return max(len(l) for l in text.split("\n"))

        srcs = {
            f: (code, r)
            for f, (code, r) in srcs.items()
            if file_width(code) <= max_line_width
        }
        result = SrcDataset(repos_root)
        result.add_stats(
            {
                "n_files_too_wide": num_all_srcs - len(srcs),
                "too_wide_ratio": (1 - len(srcs) / num_all_srcs),
                "drop_comments": drop_comments,
            }
        )
        masked_srcs: list[dict | None] = pmap(
            mask_type_annots,
            [(f, code[0]) for f, code in srcs.items()],
            [drop_comments] * len(srcs),
            max_workers=max_workers,
            desc="mask_type_annots",
            tqdm_args=tqdm_args,
        )
        filtered_srcs = []

        srcs_list = list(srcs.items())

        rands = random.getstate()
        random.seed(seed)
        for i, x in enumerate(masked_srcs):
            if x is None:
                continue
            n = len(x["types"])
            x["is_label"] = [random.random() < label_ratio for _ in range(n)]
            x["file"] = srcs_list[i][0].relative_to(repos_root)
            x["repo"] = srcs_list[i][1][1].relative_to(repos_root)
            x["prev_types"] = None
            filtered_srcs.append(x)
        random.setstate(rands)

        helper = _TokenizedSrcHelper(tokenizer)
        tk_srcs: list[TokenizedSrc] = pmap(
            helper.dict_to_tokenized_src,
            filtered_srcs,
            max_workers=max_workers,
            desc="dict_to_tokenized_src",
            tqdm_args=tqdm_args,
        )

        for f, g in groupby(tk_srcs, lambda s: s.file).items():
            assert len(g) == 1, f"{f} appears {len(g)} times."

        result.all_srcs = tk_srcs
        return result


def load_src_datasets(
    datadir: Path,
    datasets_name: str,
    data_reduction: int = 1,
    quicktest: bool = False,
    repos_root: Optional[Path] = None,
    sets_to_load=["train", "valid", "test"],
) -> dict[str, SrcDataset]:
    print("Loading datasets: ", datasets_name)
    src_datasets = dict[str, SrcDataset]()
    for n in sets_to_load:
        with open(datadir / "SPOT-data" / datasets_name / f"{n}.pkl", "rb") as f:
            src: SrcDataset = pickle.load(f)
            src = SrcDataset(src.repos_root, src.srcs_with_labels())
            if repos_root is not None:
                src.repos_root = repos_root
            if n == "train":
                n_train = len(src.all_srcs) // data_reduction
                src = src[:n_train]
            if quicktest:
                ids = range(0, len(src.all_srcs), max(1, len(src.all_srcs) // 10))
                src = src[ids]
            src_datasets[n] = src
    return src_datasets


def code_to_check_from_preds(src: TokenizedSrc, preds: dict[int, str]):
    code = src.origin_code
    changes = list[tuple[CodeRange, int, str]]()
    start = CodePosition(0, 0)
    changes.append((CodeRange(start, start), 0, MypyChecker.Preamble))
    for k in preds.keys():
        assert k in range(len(src.types)), f"Prediction index out of range: {k}"
    for i, info in enumerate(src.types_info):
        r = not_none(info.annot_range)
        pred = preds.get(i, "Any")
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


def type_check_src_skip_check(
    src: TokenizedSrc,
    preds: dict[int, str],
) -> SrcCheckResult:
    new_code = code_to_check_from_preds(src, preds)
    return SrcCheckResult([], new_code)


def type_check_src(
    src: TokenizedSrc,
    preds: dict[int, str],
    mypy_path: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> SrcCheckResult:
    def from_preds(preds: dict[int, str]):
        new_code = code_to_check_from_preds(src, preds)
        check_r = MypyChecker.check_code(new_code, cwd=cwd, mypy_path=mypy_path)
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
    preds: dict[int, str],
    project_root: Path,
    include_all_errors: bool = False,
    mypy_path: Optional[Path] = None,
) -> SrcCheckResult:
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

    rel_path = src.file.relative_to(src.repo)
    file_path = cwd / rel_path

    def from_preds(preds: dict[int, str]):
        new_code = code_to_check_from_preds(src, preds)
        file_path.write_text(new_code)
        check_r = MypyChecker.check_project(cwd, mypy_path=mypy_path)
        feedback: list[MypyFeedback] | str
        if isinstance(check_r, str):
            feedback = check_r
        else:
            feedback = check_r.error_dict.get(file_path.resolve(), [])
        return feedback, new_code

    try:
        if include_all_errors:
            fdbk, new_code = from_preds(preds)
            return SrcCheckResult(fdbk, new_code)
        else:
            fdbk0, _ = from_preds({i: "Any" for i, _ in preds.items()})
            fdbk1, new_code = from_preds(preds)
            if isinstance(fdbk1, list) and isinstance(fdbk0, list):
                preexisting = {(f.position.line, f.message) for f in fdbk0}
                new_fdbk = [
                    f for f in fdbk1 if (f.position.line, f.message) not in preexisting
                ]
                return SrcCheckResult(new_fdbk, new_code)
            else:
                return SrcCheckResult(fdbk1, new_code)
    finally:
        shutil.rmtree(MypyChecker.temp_dir() / proc.name)


class CommentRemover(cst.CSTTransformer):
    """Removes comments and docstrings."""

    def leave_IndentedBlock(
        self, node: cst.IndentedBlock, updated: cst.IndentedBlock
    ) -> cst.IndentedBlock:
        new_body = type(updated.body)(  # type: ignore
            filter(lambda n: not CommentRemover.is_doc_string(n), updated.body)
        )
        if len(new_body) != len(updated.body):
            return updated.with_changes(body=new_body)
        else:
            return updated

    def leave_Module(self, node, updated):
        return self.leave_IndentedBlock(node, updated)

    def leave_EmptyLine(self, node: cst.EmptyLine, updated: cst.EmptyLine):
        if updated.comment is not None:
            return cst.RemoveFromParent()
        else:
            return updated

    def leave_TrailingWhitespace(self, node, updated: cst.TrailingWhitespace):
        if updated.comment is not None:
            return updated.with_changes(comment=None)
        else:
            return updated

    @staticmethod
    def is_doc_string(node: cst.BaseStatement) -> bool:
        match node:
            case cst.SimpleStatementLine(body=[cst.Expr(value=cst.SimpleString())]):
                return True
            case _:
                return False


def remove_comments(m: cst.Module) -> cst.Module:
    """Removes all comments and docstrings."""
    return m.visit(CommentRemover())


def mask_type_annots(
    file_code: Union[str, tuple[Path, str]], drop_comments: bool, silent: bool = True
) -> Optional[dict]:
    """Preprocess the Python code to carve out all the type annotations. The original
    code is split into sequences at the type annotations."""

    if isinstance(file_code, tuple):
        src_path, code = file_code
    else:
        assert isinstance(file_code, str)
        src_path = Path("[no source file]")
        code = file_code
    try:
        m = cst.parse_module(code)
        if drop_comments:
            m = remove_comments(m)
    except cst.ParserSyntaxError as e:
        if not silent:
            logging.warning(f"Failed to parse src file: `{src_path}`")
        return None

    annots_info, types = collect_user_annotations(m)
    cst_code = m.code
    types_str = [
        m.code_for_node(not_none(info.annot).annotation) for info in annots_info
    ]
    mask_annot = cst.Annotation(cst.Name(SpecialNames.TypeMask))
    replaces = dict()
    for info in annots_info:
        replaces[info.path] = mask_annot
    new_code = apply_annotations(m, replaces).code
    code_segs = new_code.split(SpecialNames.TypeMask)

    assert (
        len(code_segs) == len(types) + 1
    ), f"{len(code_segs)} != {len(types) + 1}. replaces: {replaces}\ncode: {new_code}"
    return {
        "code_segs": code_segs,
        "types": types,
        "types_str": types_str,
        "annots_info": annots_info,
        "cst_code": cst_code,
    }


@dataclass
class SrcChunkInfo:
    """Stores the source code information for a chunk of tokens."""

    types: list[PythonType]  # the label types in this chunk
    annots_info: list[AnnotInfo]  # the label AnnotInfos in this chunk
    # maps each label to its source file id
    src_ids: list[int]
    # maps each label to its label id in the corresponding TokenizedSrc
    label_ids: list[int]
    prev_types: list[PythonType] | None
    inlined_spans: list[slice] | None

    def __repr__(self):
        return f"SrcChunkInfo(num_types={len(self.types)}, unique_src_ids={set(self.src_ids)})"


@dataclass
class ChunkedDataset:
    data: Dataset
    chunks_info: list[SrcChunkInfo]
    # The source files of this data set
    files: list[Path]
    file2src: dict[Path, str]
    file2repo: dict[Path, Path]
    tokenizer: TokenizerSPOT

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
            tokenizer=self.tokenizer,
        )

    def __len__(self):
        assert_eq(len(self.data), len(self.chunks_info))
        return len(self.data)

    def __repr__(self):
        return f"ChunkedDataset(num_chunks={len(self.chunks_info)}, num_srcs={len(self.files)})"

    def verify_labels(self, srcs: SrcDataset, tokenizer: TokenizerSPOT, tqdm_args={}):
        """
        Verify that the labels in the dataset match the source code.
        """

        src_path_map = dict[Path, dict[AnnotPath, PythonType]]()
        for f, src in srcs.file2src().items():
            src_path_map[f] = {
                info.path: ty for ty, info in zip(src.types, src.types_info)
            }
            assert_eq(len(src_path_map[f]), len(src.types))
        input_ids = tqdm(self.data["input_ids"], desc="verify_labels", **tqdm_args)
        for input, chunk in zip(input_ids, self.chunks_info):
            for info, ty, sid in zip(chunk.annots_info, chunk.types, chunk.src_ids):
                file = self.files[sid]
                assert file in src_path_map, f"{file} not in file2src."
                assert (
                    info.path in src_path_map[file]
                ), f"{info.path} should not be a label in {file}. Chunk code:\n{tokenizer.decode(input)}"
                assert_eq(src_path_map[file][info.path], ty)


def save_datasets(
    datasets: dict[str, ChunkedDataset],
    repos_split: dict[str, list[GitRepo]],
    datasets_dir: Path,
):
    if datasets_dir.exists():
        print("Deleting old datasets at:", datasets_dir)
        shutil.rmtree(datasets_dir)
    datasets_dir.mkdir(parents=True)

    with open(datasets_dir / "repos_split.pkl", "wb") as f:
        pickle.dump(repos_split, f)

    for name, dataset in datasets.items():
        dataset.data.save_to_disk(str(datasets_dir / name))
        extra = dataset.chunks_info, dataset.files, dataset.file2src, dataset.file2repo
        with open(datasets_dir / f"{name}-extra.pkl", "wb") as f:
            pickle.dump(extra, f)
    import subprocess

    subprocess.run(["du", "-sh", datasets_dir])


def load_datasets(datasets_dir: Path):
    set_names = ["train", "valid", "test"]
    with open(datasets_dir / "repos_split.pkl", "rb") as f:
        repos_split: dict[str, list[GitRepo]] = pickle.load(f)
    datasets = dict[str, ChunkedDataset]()
    for name in set_names:
        with open(datasets_dir / f"{name}-extra.pkl", "rb") as f:
            extra = pickle.load(f)
        dataset = Dataset.load_from_disk(str(datasets_dir / name))
        datasets[name] = ChunkedDataset(dataset, *extra)

    return datasets, repos_split


def output_ids_as_seqs(output_ids: Iterable[int], tokenizer: TokenizerSPOT):
    """Divide the model output as a sequence of tokens, filtering out padding tokens."""
    seq_id = 0
    buff = list[int]()
    seqs = list[list[int]]()
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


def output_ids_as_types(
    output_ids: Iterable[int], tokenizer: TokenizerSPOT, n_types: int
) -> list[PythonType]:
    """Try to parse model outputs as a list of Python types, pad `Any` to make sure the
    list is of the correct length."""
    seqs = output_ids_as_seqs(output_ids, tokenizer)
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


def patch_code_with_extra(
    code: str,
    predictions: dict[CodeRange, str],
    errors: dict[CodePosition, str],
    patch_predictions: bool,
) -> str:
    replaces = []
    # When the ranges overlap, we want to use the order: new_prediction -> prev_prediction -> errors
    for r, t in predictions.items():
        replaces.append((r, 1, SpecialNames.TypeMask))
        if patch_predictions:
            replaces.append((CodeRange(r.start, r.start), 2, f"/* {t} */"))

    for p, e in errors.items():
        replaces.append((CodeRange(p, p), 3, f"/* error: {e} */"))

    return replace_strs_by_pos(code, replaces)


def R1_srcs_from_preds(
    tokenizer: TokenizerSPOT,
    r0_src: SrcDataset,
    chunks_info: list[SrcChunkInfo],
    src_files: list[Path],
    r0_preds: list[list[PythonType]],
    tc_args: TypeCheckArgs,
    max_workers: int,
    tqdm_args: dict = {},
) -> SrcDataset:
    file2preds = dict[Path, dict[AnnotPath, str]]()
    for preds, chunk_info in zip(r0_preds, chunks_info):
        assert_eq(len(preds), len(chunk_info.types))
        for i, pred in enumerate(preds):
            sid = chunk_info.src_ids[i]
            file = src_files[sid]
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

    # assert_eq(len(file2preds1), len(r0_src.srcs_with_labels()))
    return r0_src.add_type_checker_feedback(
        tokenizer,
        file2preds1,
        tc_args=tc_args,
        max_workers=max_workers,
        tqdm_args=tqdm_args,
    )


def type_accuracies(
    pred_types: Sequence[PythonType],
    label_types: Sequence[PythonType],
    types_cat: Sequence[AnnotCat],
    types_pos: Sequence[int],
    normalize_types=True,
) -> dict[str, Any]:
    assert_eq(len(pred_types), len(label_types), len(types_cat), len(types_pos))

    if normalize_types:
        pred_types = [normalize_type(ty) for ty in pred_types]
        label_types = [normalize_type(ty) for ty in label_types]

    def i_to_range(i):
        if i == 0:
            return range(0, 1)
        p = int(math.log(i, 2))
        return range(2**p, 2 ** (p + 1))

    def ast_size(ty: PythonType) -> int:
        return 1 + sum(ast_size(a) for a in ty.args)

    def ast_overlap(ty1: PythonType, ty2: PythonType) -> int:
        if ty1.head != ty2.head:
            return 0
        return 1 + sum(ast_overlap(a1, a2) for a1, a2 in zip(ty1.args, ty2.args))

    partial_by_cat = GroupedAccCounter[AnnotCat]()
    partial_by_pos = GroupedAccCounter[range]()
    full_acc = GroupedAccCounter[None]()
    ast_acc = GroupedAccCounter[None]()

    for p, l, cat, pos in zip(pred_types, label_types, types_cat, types_pos):
        partial_by_cat.count(cat, p.head_name() == l.head_name(), 1)
        partial_by_pos.count(i_to_range(pos), p.head_name() == l.head_name(), 1)
        full_acc.count(None, p == l, 1)
        ast_acc.count(None, ast_overlap(p, l), ast_size(l))

    return {
        "partial_acc": partial_by_cat.overall_acc(),
        "ast_acc": ast_acc.overall_acc(),
        "full_acc": full_acc.overall_acc(),
        "partial_acc_by_cat": partial_by_cat.grouped_accs(
            key=lambda x: x.name, sort_by=lambda x: x.value
        ),
        "partial_acc_by_pos": partial_by_pos.grouped_accs(sort_by=lambda x: x.start),
        "avg_label_size": safe_div(
            sum(ast_size(l) for l in label_types), len(label_types)
        ),
        "avg_pred_size": safe_div(
            sum(ast_size(p) for p in pred_types), len(pred_types)
        ),
    }


def preds_to_accuracies(
    preds: Sequence[Sequence[PythonType]], dataset: ChunkedDataset, normalize_types=True
):
    cats = [an.cat for info in dataset.chunks_info for an in info.annots_info]
    labels = [ty for info in dataset.chunks_info for ty in info.types]
    poses = [i for info in dataset.chunks_info for i in range(len(info.types))]
    return type_accuracies(
        list(seq_flatten(preds)), labels, cats, poses, normalize_types=normalize_types
    )


def _turn_off_tokenizer_warning(tokenizer: TokenizerSPOT):
    tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True


def get_dataset_name(drop_comments: bool, all_labels: bool, spot_round: int = 0):
    drop_tag = "-drop_comments" if drop_comments else ""
    label_tag = "-all_labels" if all_labels else ""
    round_tag = f"-R{spot_round}" if spot_round > 0 else ""
    return f"src_datasets{round_tag}{label_tag}{drop_tag}"
