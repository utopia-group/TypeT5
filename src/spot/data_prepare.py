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
    mask_annot = cst.Annotation(cst.Name(f"SPOT_TYPE_MASK"))
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


def chunk_masked_code(
    srcs: Sequence[dict], chunk_size: int, tokenizer: RobertaTokenizer
) -> list[dict]:
    """
    Concatenate all the code segments into a single long sequence, then break it down 
    into even-sized chunks. Each src code is assumed to have already been processed by 
    `mask_type_annots`."""
    all_tks: list[Any] = []
    for src in srcs:
        segs: list[str] = src["code_segs"]
        types: list[PythonType] = src["types"]
        assert len(segs) == len(types) + 1
        all_tks.append(tokenizer.bos_token_id)
        for i in range(len(types)):
            all_tks.extend(tokenizer.encode(segs[i], add_special_tokens=False))
            all_tks.append(types[i])
        all_tks.extend(tokenizer.encode(segs[-1], add_special_tokens=False))
        all_tks.append(tokenizer.eos_token_id)

    # group all_tks into chunks
    chunks: list[dict] = []
    for i in range(0, len(all_tks), chunk_size):
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
                assert isinstance(tk, PythonType)
                assert extra_id <= 99, "> 99 annotations in a single sequence"
                chunk.append(tokenizer.additional_special_tokens_ids[99 - extra_id])
                types.append(tk)
                extra_id += 1
        # chunk = tokenizer.convert_ids_to_tokens(chunk) # for debugging
        assert len(chunk) == chunk_size
        label_ids = [tokenizer.bos_token_id]
        for i, ty in enumerate(types):
            label_ids.append(tokenizer.additional_special_tokens_ids[99 - i])
            label_ids.extend(tokenizer.encode(str(ty), add_special_tokens=False))
        label_ids.append(tokenizer.eos_token_id)

        chunks.append({"input_ids": chunk, "types": types, "label_ids": label_ids})
    return chunks
