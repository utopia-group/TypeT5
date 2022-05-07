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
from transformers import RobertaTokenizer

from spot.type_env import (
    AnnotCat,
    AnnotInfo,
    AnnotPath,
    PythonType,
    apply_annotations,
    collect_annotations,
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


def mask_type_annots(file_code: Union[str, tuple[Path, str]]) -> Optional[dict]:
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
        logging.warning(f"Failed to parse src file: `{src_path}`")
        return None
    paths = collect_annotations(m)
    types: list[PythonType] = []
    annots_info: list[AnnotInfo] = []
    replaces = dict()
    label_id = 0
    mask_annot = cst.Annotation(cst.Name("SPOT_TYPE_MASK"))
    for info in paths:
        if info.annot is None:
            continue
        p = info.path
        ty = parse_type_expr(m, info.annot.annotation, silent=True)
        if ty is not None:
            types.append(ty)
            annots_info.append(info)
            replaces[p] = mask_annot
            label_id += 1
    m1 = apply_annotations(m, replaces)
    code_segs = m1.code.split("SPOT_TYPE_MASK")
    return {
        "code_segs": code_segs,
        "types": types,
        "annots_info": annots_info,
    }


def tokenize_masked(masked: dict, tokenizer: RobertaTokenizer, device):
    mask_tokens = [f"<extra_id_{i}>" for i in range(len(masked["types"]))]
    input_ids = tokenizer.encode(
        join_str(masked["code_segs"], mask_tokens), return_tensors="pt"
    )
    label_str = "".join(a + str(b) for a, b in zip(mask_tokens, masked["types"]))
    labels = tokenizer.encode(label_str, return_tensors="pt")
    return {"input_ids": input_ids.to(device), "labels": labels.to(device)}


def chunk_masked_code(
    srcs: Sequence[dict],
    chunk_size: int,
    tokenizer: RobertaTokenizer,
    ctx_margin=None,
):
    """
    Concatenate all the code segments into a single long sequence, then break it down
    into even-sized chunks. Each source code is assumed to have already been processed by
    `mask_type_annots`.

    Args:
        ctx_margin: the number of tokens on each end of the chunk that are not masked. Only
        the `chunk_size - 2 * ctx_margin` middle tokens in the chunk are masked.
    """

    def expand_types_as_tks(mixed_tks: list):
        result = list[int]()
        for e in mixed_tks:
            if isinstance(e, int):
                result.append(e)
            else:
                assert isinstance(e[0], PythonType)
                type_tks = tokenizer.encode(str(e[0]), add_special_tokens=False)
                result.extend(type_tks)
        return result

    if ctx_margin is None:
        ctx_margin = chunk_size // 4
    stride = chunk_size - 2 * ctx_margin

    all_tks: list[Any] = []
    for src in tqdm(srcs, desc="Concatinating all srcs"):
        segs: list[str] = src["code_segs"]
        types_labels: list[PythonType] = src["types"]
        types_cat: list[AnnotCat] = src["types_cat"]
        assert (
            len(segs) == len(types_labels) + 1
        ), f"len(segs)={len(segs)}, len(types_labels)={len(types_labels)}"
        all_tks.append(tokenizer.bos_token_id)
        for i in range(len(types_labels)):
            all_tks.extend(tokenizer.encode(segs[i], add_special_tokens=False))
            all_tks.append((types_labels[i], types_cat[i]))
        all_tks.extend(tokenizer.encode(segs[-1], add_special_tokens=False))
        all_tks.append(tokenizer.eos_token_id)

    # slide over `all_tks` with step size `stride` to turn them into masked chunks
    chunks: dict[str, list] = {
        "input_ids": [],
        "labels": [],
        "types": [],
        "types_cat": [],
    }
    for i in tqdm(range(0, len(all_tks), stride), desc="Chunking"):
        tks = all_tks[i : i + chunk_size]
        if len(tks) != chunk_size:
            continue  # too short, discard
        extra_id = 0
        middle = []
        types = []
        types_cat = []

        for tk in tks[ctx_margin : ctx_margin + stride]:
            if isinstance(tk, int):
                middle.append(tk)
            else:
                assert isinstance(
                    tk[0], PythonType
                ), f"unexpected token type {type(tk)}: {tk}"
                assert extra_id <= 99, "> 99 annotations in a single sequence"
                middle.append(tokenizer.additional_special_tokens_ids[99 - extra_id])
                types.append(str(tk[0]))
                types_cat.append(tk[1].value)
                extra_id += 1

        if len(types) == 0:
            continue  # no types to predict in this chunk, discard
        left_ctx = expand_types_as_tks(tks[:ctx_margin])[-ctx_margin:]
        right_ctx = expand_types_as_tks(tks[-ctx_margin:])[:ctx_margin]
        input_ids = left_ctx + middle + right_ctx
        assert len(input_ids) == chunk_size

        label_ids = [tokenizer.bos_token_id]
        for i, ty in enumerate(types):
            label_ids.append(tokenizer.additional_special_tokens_ids[99 - i])
            label_ids.extend(tokenizer.encode(str(ty), add_special_tokens=False))
        label_ids.append(tokenizer.eos_token_id)

        chunks["input_ids"].append(input_ids)
        chunks["labels"].append(label_ids)
        chunks["types"].append(types)
        chunks["types_cat"].append(types_cat)
    return chunks


def repos_to_dataset(
    repos: Sequence[GitRepo],
    repos_dir,
    tokenizer: RobertaTokenizer,
    max_workers: int,
    ctx_margin: Optional[int],
):
    tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True
    srcs = [
        (f, code)
        for r in tqdm(repos, desc="read_files")
        for (f, code) in r.src_files(repos_dir)
    ]
    chunk_size = max(1, min(50, len(srcs) // max_workers))
    masked_srcs = process_map(
        mask_type_annots,
        srcs,
        max_workers=max_workers,
        desc="tokenize sources",
        chunksize=chunk_size,
    )
    # filter out srcs that failed to parse
    masked_srcs = [x for x in masked_srcs if x is not None]
    logging.info(f"{len(masked_srcs)} / {len(srcs)} srcs succesfully parsed.")
    context_len = tokenizer.model_max_length
    return Dataset.from_dict(
        chunk_masked_code(masked_srcs, context_len, tokenizer, ctx_margin)
    )


def load_or_process_datasets(
    datasets_dir: Path,
    tokenizer: RobertaTokenizer,
    repos_dir,
    repos_train: Sequence[GitRepo],
    repos_valid: Sequence[GitRepo],
    repos_test: Sequence[GitRepo],
    max_workers: int = 12,
    regenerate: bool = False,
    ctx_margin: Optional[int] = None,
):
    tk_datasets: dict[str, Dataset]
    repos_split: dict[str, Sequence[GitRepo]]

    set_names = ["train", "valid", "test"]
    if datasets_dir.exists() and not regenerate:
        print("Loading datasets from:", datasets_dir)
        with open(datasets_dir / "repos_split.pkl", "rb") as f:
            repos_split = pickle.load(f)
        tk_datasets = {
            name: Dataset.load_from_disk(str(datasets_dir / name)) for name in set_names
        }
        return tk_datasets, repos_split
    # need to generate datasets
    if datasets_dir.exists():
        print("Deleting old datasets at:", datasets_dir)
        shutil.rmtree(datasets_dir)
    repos_split = {"train": repos_train, "valid": repos_valid, "test": repos_test}
    datasets_dir.mkdir(parents=True)
    with open(datasets_dir / "repos_split.pkl", "wb") as f:
        pickle.dump(repos_split, f)

    tk_datasets = dict()
    for name, repos in repos_split.items():
        print(f"Processing dataset: {name}")
        d = repos_to_dataset(
            repos, repos_dir, tokenizer, max_workers, ctx_margin=ctx_margin
        )
        tk_datasets[name] = d
        d.save_to_disk(str(datasets_dir / name))
    return tk_datasets, repos_split


def output_ids_as_seqs(output_ids: list[int], tokenizer: RobertaTokenizer):
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


def output_ids_as_types(output_ids: list[int], tokenizer: RobertaTokenizer):
    """Try to parse model outputs as a list of Python types."""
    seqs = output_ids_as_seqs(output_ids, tokenizer)
    types = list[PythonType]()
    for seq in seqs:
        try:
            ex_str = tokenizer.decode(seq, skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Failed to decode sequence: {seq}") from e
        try:
            tree = ast.parse(ex_str, mode="eval").body
            ty = parse_type_from_ast(tree)
        except:
            ty = PythonType.Any()
        types.append(ty)
    return types


class ModuleRemapUnpickler(pickle.Unpickler):
    def __init__(self, file, module_map, **kw_args) -> None:
        self.module_map: Callable[[str], str] = module_map
        super().__init__(file, **kw_args)

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(self.module_map(module), name)
