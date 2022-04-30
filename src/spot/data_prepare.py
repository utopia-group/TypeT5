import shutil
import subprocess
from types import NoneType
from dataclasses import dataclass
from transformers import RobertaTokenizer
from spot.type_env import (
    AnnotPath,
    PythonType,
    apply_annotations,
    collect_annotations,
    parse_type_expr,
)
from spot.utils import *
from typing import *
from datetime import datetime
import dateparser
import warnings
import logging
from datasets import Dataset
import pickle

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

    def repo_dir(self, repos_dir):
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
        self.last_update = dateparser.parse(s.split("+")[0]).replace(tzinfo=None)
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

    def collect_annotations(self, repos_dir, silent=True):
        n_paths, n_annots = 0, 0
        file_to_annots = dict[Path, dict[AnnotPath, PythonType | None]]()
        for src in self.repo_dir(repos_dir).glob("**/*.py"):
            src: Path
            rpath = src.relative_to(self.repo_dir(repos_dir))
            m = cst.parse_module(read_file(src))
            paths, annots = collect_annotations(m)
            n_paths += len(paths)
            n_annots += len(annots)
            file_to_annots[rpath] = {
                k: parse_type_expr(m, v.annotation, silent) for k, v in annots.items()
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


def mask_type_annots(code: str):
    """Preprocess the Python code to carve out all the type annotations. The original
    code is split into sequences at the type annotations."""
    m = cst.parse_module(code)
    paths, truths = collect_annotations(m)
    types: list[PythonType] = []
    replaces = dict()
    label_id = 0
    mask_annot = cst.Annotation(cst.Name("SPOT_TYPE_MASK"))
    for p in paths:
        if p in truths:
            ty = parse_type_expr(m, truths[p].annotation, silent=True)
            if ty is not None:
                types.append(ty)
                replaces[p] = mask_annot
                label_id += 1
    m1 = apply_annotations(m, replaces)
    code_segs = m1.code.split("SPOT_TYPE_MASK")
    return {"code_segs": code_segs, "types": types}


def tokenize_masked(masked: dict, tokenizer: RobertaTokenizer, device):
    mask_tokens = [f"<extra_id_{i}>" for i in range(len(masked["types"]))]
    input_ids = tokenizer.encode(
        join_str(masked["code_segs"], mask_tokens), return_tensors="pt"
    )
    label_str = "".join(a + str(b) for a, b in zip(mask_tokens, masked["types"]))
    labels = tokenizer.encode(label_str, return_tensors="pt")
    return {"input_ids": input_ids.to(device), "labels": labels.to(device)}


def chunk_masked_code(
    srcs: Sequence[dict], chunk_size: int, tokenizer: RobertaTokenizer
):
    """
    Concatenate all the code segments into a single long sequence, then break it down
    into even-sized chunks. Each src code is assumed to have already been processed by
    `mask_type_annots`."""
    all_tks: list[Any] = []
    for src in tqdm(srcs, desc="Concatinating all srcs"):
        segs: list[str] = src["code_segs"]
        types_labels: list[PythonType] = src["types"]
        assert len(segs) == len(types_labels) + 1
        all_tks.append(tokenizer.bos_token_id)
        for i in range(len(types_labels)):
            all_tks.extend(tokenizer.encode(segs[i], add_special_tokens=False))
            all_tks.append(types_labels[i])
        all_tks.extend(tokenizer.encode(segs[-1], add_special_tokens=False))
        all_tks.append(tokenizer.eos_token_id)

    # group all_tks into chunks
    chunks: dict[str, list] = {"input_ids": [], "labels": [], "types": []}
    for i in tqdm(range(0, len(all_tks), chunk_size), desc="Chunking"):
        tks = all_tks[i : i + chunk_size]
        if len(tks) != chunk_size:
            continue  # discard
        extra_id = 0
        chunk = []
        types = []
        for tk in tks:
            if isinstance(tk, int):
                chunk.append(tk)
            else:
                assert isinstance(
                    tk, PythonType
                ), f"unexpected token type {type(tk)}: {tk}"
                assert extra_id <= 99, "> 99 annotations in a single sequence"
                chunk.append(tokenizer.additional_special_tokens_ids[99 - extra_id])
                types.append(str(tk))
                extra_id += 1
        # chunk = tokenizer.convert_ids_to_tokens(chunk) # for debugging
        assert len(chunk) == chunk_size
        label_ids = [tokenizer.bos_token_id]
        for i, ty in enumerate(types):
            label_ids.append(tokenizer.additional_special_tokens_ids[99 - i])
            label_ids.extend(tokenizer.encode(str(ty), add_special_tokens=False))
        label_ids.append(tokenizer.eos_token_id)

        chunks["input_ids"].append(chunk)
        chunks["labels"].append(label_ids)
        chunks["types"].append(types)
    return chunks


def repos_to_dataset(
    repos: Sequence[GitRepo],
    repos_dir,
    tokenizer: RobertaTokenizer,
    max_workers: int,
):
    tokenizer.deprecation_warnings[
        "sequence-length-is-longer-than-the-specified-maximum"
    ] = True
    srcs = [
        code
        for r in tqdm(repos, desc="read_files")
        for (_, code) in r.src_files(repos_dir)
    ]
    chunk_size = min(50, len(srcs) // max_workers)
    masked_srcs = process_map(
        mask_type_annots,
        srcs,
        max_workers=max_workers,
        desc="tokenize sources",
        chunksize=chunk_size,
    )
    context_len = tokenizer.model_max_length
    return Dataset.from_dict(chunk_masked_code(masked_srcs, context_len, tokenizer))


def load_or_process_datasets(
    datasets_dir: Path,
    tokenizer: RobertaTokenizer,
    repos_dir,
    repos_train: Sequence[GitRepo],
    repos_valid: Sequence[GitRepo],
    repos_test: Sequence[GitRepo],
    max_workers: int = 12,
    regenerate: bool = False,
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
        d = repos_to_dataset(repos, repos_dir, tokenizer, max_workers)
        tk_datasets[name] = d
        d.save_to_disk(str(datasets_dir / name))
    return tk_datasets, repos_split
