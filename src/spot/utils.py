from typing import List, Tuple, Dict, Set, Optional, Union
from typing import cast
import libcst as cst
import os
from pathlib import Path

class SpecialNames:
    Return = "<return>"

def read_file(path: str) -> str:
    """read file content as string."""
    with open(path, "r") as f:
        return f.read()

def write_file(path: str, content: str) -> None:
    """write content to file."""
    with open(path, "w") as f:
        f.write(content)

def test_f():
    return 2