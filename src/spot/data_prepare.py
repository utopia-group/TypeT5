import subprocess
from types import NoneType
from dataclasses import dataclass
from spot.type_checking import collect_annotations
from spot.utils import *
from typing import *
from datetime import datetime
import dateparser
import warnings

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

    def count_annotations(self, repos_dir) -> None:
        n_paths, n_annots = 0, 0
        for src in self.repo_dir(repos_dir).glob("**/*.py"):
            m = cst.parse_module(read_file(src))
            paths, annots = collect_annotations(m)
            n_paths += len(paths)
            n_annots += len(annots)
        self.n_type_annots = n_annots
        self.n_type_places = n_paths

    @staticmethod
    def from_json(json):
        return GitRepo(
            author=json["author"],
            name=json["repo"],
            url=json["repoUrl"],
            stars=json["stars"],
            forks=json["forks"],
        )
