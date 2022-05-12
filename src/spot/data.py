import logging
import pickle
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import *

import dateparser
from datasets import Dataset

from spot.model import TokenizerSPOT
from spot.type_env import (
    AnnotCat,
    AnnotInfo,
    AnnotPath,
    PythonType,
    apply_annotations,
    collect_annotations,
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
            paths = collect_annotations(m)
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


def mask_type_annots(
    file_code: Union[str, tuple[Path, str]], silent: bool = True
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
    except cst.ParserSyntaxError as e:
        if not silent:
            logging.warning(f"Failed to parse src file: `{src_path}`")
        return None
    paths = collect_annotations(m)
    types: list[PythonType] = []
    annots_info: list[AnnotInfo] = []
    replaces = dict()
    mask_annot = cst.Annotation(cst.Name(SpecialNames.TypeMask))
    for info in paths:
        if info.annot is None:
            continue
        ty = parse_type_expr(m, info.annot.annotation, silent=True)
        # Wierd: check if ty is None does not always work when using multiprocessing.
        if hasattr(ty, "head") and hasattr(ty, "args"):
            types.append(cast(PythonType, ty))
            annots_info.append(info)
            replaces[info.path] = mask_annot
    m1 = apply_annotations(m, replaces)
    code_segs = m1.code.split(SpecialNames.TypeMask)
    assert (
        len(code_segs) == len(types) + 1
    ), f"{len(code_segs)} != {len(types) + 1}. replaces: {replaces}\ncode: {m1.code}"
    return {
        "code_segs": code_segs,
        "types": types,
        "annots_info": annots_info,
        "code": m.code,
    }


# Todo: deprecate this
def tokenize_masked(masked: dict, tokenizer: TokenizerSPOT, device):
    mask_tokens = [f"<extra_id_{i}>" for i in range(len(masked["types"]))]
    input_ids = tokenizer.encode(
        join_str(masked["code_segs"], mask_tokens), return_tensors="pt"
    )
    label_str = "".join(a + str(b) for a, b in zip(mask_tokens, masked["types"]))
    labels = tokenizer.encode(label_str, return_tensors="pt")
    return {"input_ids": input_ids.to(device), "labels": labels.to(device)}


def _tokenize_masked_code(
    src: dict, src_id: int, tokenizer: TokenizerSPOT
) -> list[int | tuple]:
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    assert bos_id is not None
    assert eos_id is not None

    all_tks: list[int | tuple] = []
    segs: list[str] = src["code_segs"]
    types_labels: list[PythonType] = src["types"]
    types_info: list[AnnotInfo] = src["annots_info"]
    assert (
        len(segs) == len(types_labels) + 1
    ), f"len(segs)={len(segs)}, len(types_labels)={len(types_labels)}"
    all_tks.append(bos_id)
    for i in range(len(types_labels)):
        all_tks.extend(tokenizer.encode(segs[i], add_special_tokens=False))
        ty = types_labels[i]
        all_tks.append((ty, types_info[i], src_id))
    all_tks.extend(tokenizer.encode(segs[-1], add_special_tokens=False))
    all_tks.append(eos_id)
    return all_tks


def _process_chunk(
    tks: list[int | tuple],
    chunk_size,
    ctx_margin: int,
    tokenizer: TokenizerSPOT,
    types_in_ctx: bool,  # whether to expand the label types in the context. If not, will replace them with <mask>.
) -> Optional[dict]:
    def expand_types_as_tks(mixed_tks: list):
        result = list[int]()
        mask_id = tokenizer.mask_token_id
        assert mask_id is not None
        for e in mixed_tks:
            if isinstance(e, int):
                result.append(e)
            else:
                if types_in_ctx:
                    type_tks = tokenizer.encode(str(e[0]), add_special_tokens=False)
                    result.extend(type_tks)
                else:
                    result.append(mask_id)
        return result

    stride = chunk_size - 2 * ctx_margin
    if len(tks) != chunk_size:
        # add padding
        tks.extend([cast(int, tokenizer.pad_token_id)] * (chunk_size - len(tks)))
    extra_id = 0
    middle = []
    types = list[PythonType]()
    annots_info = list[AnnotInfo]()
    src_ids = list[int]()

    for tk in tks[ctx_margin : ctx_margin + stride]:
        if isinstance(tk, int):
            middle.append(tk)
        else:
            ty, info, src_id = tk
            assert extra_id <= 99, "> 99 annotations in a single sequence"
            middle.append(tokenizer.additional_special_tokens_ids[99 - extra_id])
            types.append(ty)
            annots_info.append(info)
            src_ids.append(src_id)
            extra_id += 1
    if extra_id == 0:
        return None  # no types to predict in this chunk, discard
    left_ctx = expand_types_as_tks(tks[:ctx_margin])[-ctx_margin:]
    right_ctx = expand_types_as_tks(tks[-ctx_margin:])[:ctx_margin]
    input_ids = left_ctx + middle + right_ctx
    assert len(input_ids) == chunk_size

    label_ids = [tokenizer.bos_token_id]
    for i, ty in enumerate(types):
        label_ids.append(tokenizer.additional_special_tokens_ids[99 - i])
        label_ids.extend(tokenizer.encode(str(ty), add_special_tokens=False))
    label_ids.append(tokenizer.eos_token_id)
    meta = SrcChunkInfo(types, annots_info, src_ids)

    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "meta": meta,
    }


@dataclass
class _ChunkingHelper:
    """Multi-process helper for `chunk_masked_code`."""

    tokenizer: TokenizerSPOT
    chunk_size: int
    ctx_margin: int
    types_in_ctx: bool

    def tokenize(self, src: tuple[int, dict]):
        return _tokenize_masked_code(src[1], src[0], self.tokenizer)

    def process_chunk(self, tks: list[int | tuple]):
        return _process_chunk(
            tks, self.chunk_size, self.ctx_margin, self.tokenizer, self.types_in_ctx
        )


@dataclass
class SrcChunkInfo:
    """Stores the source code information for a chunk of tokens."""

    types: list[PythonType]  # the label types in this chunk
    annots_info: list[AnnotInfo]  # the label AnnotInfos in this chunk
    # maps each label to its source file id
    src_ids: list[int]


def chunk_masked_code(
    srcs: Sequence[dict],
    tokenizer: TokenizerSPOT,
    chunk_size: int,
    ctx_margin: Optional[int],
    types_in_ctx: bool,
    max_workers: int,
    *,
    silent=False,
) -> tuple[Dataset, list[SrcChunkInfo]]:
    """
    Concatenate all the code segments into a single long sequence, then break it down
    into even-sized chunks. Each source code is assumed to have already been processed by
    `mask_type_annots`.

    Args:
        ctx_margin: the number of tokens on each end of the chunk that are not masked. Only
        the `chunk_size - 2 * ctx_margin` middle tokens in the chunk are masked.
    """

    def pmap(f, xs: Sequence, desc: str):
        chunksize = max(len(xs) // (8 * max_workers), 1)
        return process_map(
            f,
            xs,
            max_workers=max_workers,
            chunksize=chunksize,
            desc=desc,
            disable=silent,
        )

    if ctx_margin is None:
        ctx_margin = chunk_size // 4

    helper = _ChunkingHelper(tokenizer, chunk_size, ctx_margin, types_in_ctx)

    # first, tokenize and concat all files into a single long sequence
    file_tks: list[list[int | tuple]] = pmap(
        helper.tokenize,
        list(enumerate(srcs)),
        desc="tokenizing sources",
    )
    all_tks: list[int | tuple] = list(seq_flatten(file_tks))

    # the, use a sliding window over `all_tks` with step size `stride` to turn them into masked chunks
    stride = chunk_size - 2 * ctx_margin
    chunk_outputs = pmap(
        helper.process_chunk,
        [all_tks[i : i + chunk_size] for i in range(0, len(all_tks), stride)],
        desc="processing chunks",
    )

    chunks: dict[str, list] = {
        "input_ids": [],
        "labels": [],
    }
    chunks_info: list[SrcChunkInfo] = []

    for chunk in chunk_outputs:
        if chunk is None:
            continue
        chunks["input_ids"].append(chunk["input_ids"])
        chunks["labels"].append(chunk["labels"])
        meta = chunk["meta"]
        chunks_info.append(meta)

    return Dataset.from_dict(chunks), chunks_info


@dataclass
class TypeInfDataset:
    data: Dataset
    chunks_info: list[SrcChunkInfo]
    # The source files of this data set
    files: list[Path]
    # The source contents seen by the parser. All code locations refer to
    # locations in these contents, which can be different from the original source files.
    srcs: dict[Path, str]

    def __getitem__(self, key: slice) -> "TypeInfDataset":
        assert isinstance(key, slice)

        new_data = {n: self.data[n][key] for n in self.data.column_names}
        new_info = self.chunks_info[key]
        return TypeInfDataset(
            Dataset.from_dict(new_data),
            chunks_info=new_info,
            files=self.files,
            srcs=self.srcs,
        )


def repos_to_dataset(
    repos_paths: Iterable[Path],
    tokenizer: TokenizerSPOT,
    max_workers: int,
    ctx_margin: Optional[int],
    types_in_ctx: bool,
    *,
    silent=False,
) -> TypeInfDataset:
    tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True
    srcs: list[tuple[Path, str]] = [
        (f, f.read_text()) for p in repos_paths for f in sorted(p.glob("**/*.py"))
    ]
    chunk_size = max(1, len(srcs) // (10 * max_workers))
    if max_workers <= 1:
        map_r = map(mask_type_annots, srcs)
    else:
        map_r = process_map(
            mask_type_annots,
            srcs,
            max_workers=max_workers,
            desc="parsing and masking sources",
            chunksize=chunk_size,
            disable=silent,
        )
    # filter out srcs that failed to parse
    parsed_files = list[Path]()
    file_to_src = dict[Path, str]()
    masked_srcs = list[dict]()
    for f, src in zip(srcs, map_r):
        if src is not None:
            parsed_files.append(f[0])
            file_to_src[f[0]] = src.pop("code")
            masked_srcs.append(src)
    if not silent:
        logging.info(f"{len(masked_srcs)} / {len(srcs)} srcs succesfully parsed.")
    context_len = tokenizer.model_max_length
    dataset, chunks_info = chunk_masked_code(
        masked_srcs,
        tokenizer,
        context_len,
        ctx_margin=ctx_margin,
        max_workers=max_workers,
        types_in_ctx=types_in_ctx,
        silent=silent,
    )

    return TypeInfDataset(dataset, chunks_info, parsed_files, file_to_src)


def load_or_process_datasets(
    datasets_dir: Path,
    tokenizer: TokenizerSPOT,
    repos_dir,
    repos_train: Sequence[GitRepo],
    repos_valid: Sequence[GitRepo],
    repos_test: Sequence[GitRepo],
    max_workers: int = 12,
    regenerate: bool = False,
    ctx_margin: Optional[int] = None,
    types_in_ctx: bool = False,
):
    if datasets_dir.exists() and not regenerate:
        print("Loading datasets from:", datasets_dir)
        return load_datasets(datasets_dir)

    repos_split: dict[str, Sequence[GitRepo]]

    if datasets_dir.exists():
        print("Deleting old datasets at:", datasets_dir)
        shutil.rmtree(datasets_dir)
    repos_split = {"train": repos_train, "valid": repos_valid, "test": repos_test}
    datasets_dir.mkdir(parents=True)

    with open(datasets_dir / "repos_split.pkl", "wb") as f:
        pickle.dump(repos_split, f)

    datasets = dict()
    for name, repos in repos_split.items():
        print(f"Processing dataset: {name}")
        repo_paths = [repo.repo_dir(repos_dir) for repo in repos]
        tdata = repos_to_dataset(
            repo_paths,
            tokenizer,
            max_workers,
            ctx_margin=ctx_margin,
            types_in_ctx=types_in_ctx,
        )
        datasets[name] = tdata

        tdata.data.save_to_disk(str(datasets_dir / name))
        extra = tdata.chunks_info, tdata.files, tdata.srcs
        with open(datasets_dir / f"{name}-extra.pkl", "wb") as f:
            pickle.dump(extra, f)

    return datasets, repos_split


def load_datasets(datasets_dir: Path):
    set_names = ["train", "valid", "test"]
    with open(datasets_dir / "repos_split.pkl", "rb") as f:
        repos_split = pickle.load(f)
    datasets = dict()
    for name in set_names:
        with open(datasets_dir / f"{name}-extra.pkl", "rb") as f:
            extra = pickle.load(f)
        dataset = Dataset.load_from_disk(str(datasets_dir / name))
        datasets[name] = TypeInfDataset(dataset, *extra)

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
        assert isinstance(
            ty, PythonType
        ), f"{ty} of type {type(ty)} is not a PythonType."
        types.append(ty)
    types.extend(PythonType.Any() for _ in range(n_types - len(types)))
    assert len(types) == n_types
    return types


def patch_code_with_extra(
    code: str, predictions: dict[CodeRange, str], errors: dict[CodePosition, str]
) -> str:
    replaces = []
    first_line = "/* type checked */"
    replaces.append((CodeRange(CodePosition(1, 1), CodePosition(1, 1)), first_line))
    for r, t in predictions.items():
        replaces.append((CodeRange(r.start, r.start), f"/* {t} */"))
        replaces.append((r, SpecialNames.TypeMask))
    for p, e in errors.items():
        replaces.append((CodeRange(p, p), f"/* error: {e} */"))
    return replace_strs_by_pos(code, replaces)


def compute_metrics(
    predictions: np.ndarray,
    label_ids: np.ndarray,
    cats: list[AnnotCat],
    n_labels: Sequence[int],
    tokenizer: TokenizerSPOT,
) -> dict[str, Any]:
    # apply the tokenizer decoder to each rows
    assert len(predictions.shape) == 2
    assert (n_rows := predictions.shape[0]) == label_ids.shape[0]
    preds = list[PythonType]()
    labels = list[PythonType]()
    for i in tqdm(range(n_rows), desc="decoding types"):
        pred = output_ids_as_types(predictions[i, :], tokenizer, n_labels[i])
        label = output_ids_as_types(label_ids[i, :], tokenizer, n_labels[i])
        preds.extend(map(normalize_type, pred))
        labels.extend(map(normalize_type, label))

    r = type_accuracies(preds, labels, cats, normalize_types=False)
    r["pred_types"] = [ty.head_name() for ty in preds]
    r["label_types"] = [ty.head_name() for ty in labels]
    return r


def type_accuracies(
    pred_types: Sequence[PythonType],
    label_types: Sequence[PythonType],
    types_cat: Sequence[AnnotCat],
    normalize_types=True,
) -> dict[str, Any]:
    assert len(pred_types) == len(
        label_types
    ), f"{len(pred_types)} != {len(label_types)}"

    if normalize_types:
        pred_types = [normalize_type(ty) for ty in pred_types]
        label_types = [normalize_type(ty) for ty in label_types]

    n_correct_by_cat = Counter[AnnotCat]()
    n_partial_by_cat = Counter[AnnotCat]()
    n_label_by_cat = Counter[AnnotCat](types_cat)

    for p, l, cat in zip(pred_types, label_types, types_cat):
        if p == l:
            n_correct_by_cat[cat] += 1
        if p.head_name() == l.head_name():
            n_partial_by_cat[cat] += 1

    accuracy_partial = {"total": n_partial_by_cat.total() / n_label_by_cat.total()}
    for k in n_partial_by_cat.keys():
        accuracy_partial[k.name] = n_partial_by_cat[k] / n_label_by_cat[k]

    accuracy_full = {"total": n_correct_by_cat.total() / n_label_by_cat.total()}
    for k in n_correct_by_cat.keys():
        accuracy_full[k.name] = n_correct_by_cat[k] / n_label_by_cat[k]

    return {
        "accuracy_partial": accuracy_partial,
        "accuracy_full": accuracy_full,
        "n_labels": n_label_by_cat.total(),
    }


def inline_predictions(
    input_tks: Sequence[int],
    predictions: Sequence[Sequence[int]],
    tokenizer: TokenizerSPOT,
) -> list[int]:
    """Inline the model predictions into the input code and then decode"""
    out_tks = list[int]()
    extra_id = 0
    next_special = tokenizer.additional_special_tokens_ids[99 - extra_id]
    for tk in input_tks:
        out_tks.append(tk)
        if tk == next_special:
            out_tks.extend(predictions[extra_id])
            extra_id += 1
            next_special = tokenizer.additional_special_tokens_ids[99 - extra_id]
    assert extra_id == len(predictions), f"{extra_id} != {len(predictions)}"
    return out_tks
