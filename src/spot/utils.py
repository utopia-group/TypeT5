from typing import List, Tuple, Dict, Set, Optional, Union
from typing import cast
import libcst as cst
import os
from pathlib import Path

class SpecialNames:
    Return = "<return>"
    Missing = "<missing>"

def read_file(path) -> str:
    """read file content as string."""
    with open(path, "r") as f:
        return f.read()

def write_file(path, content: str) -> None:
    """write content to file."""
    with open(path, "w") as f:
        f.write(content)

def proj_root() -> Path:
    return Path(__file__).parent.parent.parent