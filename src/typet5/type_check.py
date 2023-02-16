import ast
import multiprocessing
import re
import shutil
import subprocess

from .utils import *


@dataclass(unsafe_hash=True, order=True)
class PythonType:
    head: tuple[str, ...]
    args: tuple["PythonType", ...] = ()

    def __str__(self):
        h = ".".join(self.head)
        out: str
        if h.startswith("<"):
            match h:
                case "<List>":
                    out = f"[{', '.join(map(str, self.args))}]"
                case "<|>":
                    out = " | ".join(map(str, self.args))
                case "<FuncCall>":
                    out = "_FuncCall_"
                case "<Tuple>":
                    if len(self.args) == 1:
                        out = f"({str(self.args[0])},)"
                    else:
                        out = f"({', '.join(map(str, self.args))})"
                case _:
                    raise ValueError(f"Don't know how to handle special head: '{h}'")
        elif len(self.args) == 0:
            out = h
        else:
            out = f"{h}[{', '.join(map(str, self.args))}]"
        return out

    def __repr__(self):
        return f"ty'{str(self)}'"

    def all_heads(self):
        """Return an iterator of all the type heads."""
        yield self.head
        for arg in self.args:
            yield from arg.all_heads()

    def all_names(self):
        yield self.head_name()
        for arg in self.args:
            yield from arg.all_names()

    def head_name(self) -> str:
        """Return the last part of the type head."""
        if self.head == ():
            return SpecialNames.Empty
        else:
            return self.head[-1]

    def is_any(self) -> bool:
        return self.head_name() == "Any"

    def is_none(self) -> bool:
        return self.head_name() == "None"

    def is_union(self) -> bool:
        """Check whether the type is a union type."""
        return self.head_name() == "Union" or self.head_name() == "<|>"

    def is_optional(self) -> bool:
        return self.head_name() == "Optional"

    def normalized(self) -> "PythonType":
        return normalize_type(self)

    @staticmethod
    def from_name(name: str) -> "PythonType":
        return PythonType((name,))

    @staticmethod
    def from_str(s: str) -> "PythonType":
        return parse_type_str(s)

    @staticmethod
    def Any() -> "PythonType":
        return PythonType.from_name("Any")


_type_name_map = {
    "list": "List",
    "tuple": "Tuple",
    "dict": "Dict",
    "set": "Set",
}


def normalize_type_name(name: str) -> str:
    return _type_name_map.get(name, name)


def normalize_type_head(head: tuple[str, ...]) -> tuple[str, ...]:
    n = len(head)
    if n == 0:
        return head
    return (*head[0 : n - 1], normalize_type_name(head[n - 1]))


def normalize_type(typ: PythonType) -> PythonType:
    n_args = tuple(map(normalize_type, typ.args))
    if typ.is_union() or typ.is_optional():
        arg_set = set[PythonType]()
        if typ.is_optional():
            arg_set.add(PythonType(("None",)))
            if len(typ.args) == 0:
                arg_set.add(PythonType.Any())
        for arg in n_args:
            if arg.is_union():
                arg_set.update(arg.args)
            else:
                arg_set.add(arg)
        union_args = tuple(sorted(arg_set))
        return PythonType(("Union",), union_args)
    if all(a.is_any() for a in n_args):
        # if all arguments are Any, we can drop them all
        n_args = tuple()

    return PythonType(normalize_type_head(typ.head), n_args)


def remove_top_optional(t: PythonType) -> PythonType:
    """
    Remove the top-level Optional. i.e., convert Optional[T] to T.
    """
    if t.is_optional():
        if len(t.args) == 1:
            return t.args[0]
        else:
            return PythonType.Any()
    elif t.is_union():
        new_args = tuple(a for a in t.args if not a.is_none())
        if len(new_args) == 1:
            return new_args[0]
        else:
            return PythonType(("Union",), tuple(new_args))
    else:
        return t


def remove_top_final(t: PythonType) -> PythonType:
    """
    Remove the top-level Final. i.e., convert Final[T] to T.
    """
    if t.head_name() == "Final":
        if len(t.args) == 1:
            return t.args[0]
        else:
            return PythonType.Any()
    else:
        return t


def remove_type_namespace(typ: PythonType) -> PythonType:
    """
    Remove the namespace from the type. i.e., convert typing.List[T] to List[T].
    """
    new_args = tuple(map(remove_type_namespace, typ.args))
    new_head = (typ.head[-1],) if typ.head else ()
    return PythonType(new_head, new_args)


def limit_type_depth(typ: PythonType, max_depth: int) -> PythonType:
    """
    Limit the depth of the type to max_depth.
    """
    if max_depth <= 0:
        return PythonType.Any()
    new_args = tuple(map(lambda t: limit_type_depth(t, max_depth - 1), typ.args))
    return PythonType(typ.head, new_args)


def parse_type_str(typ_str: str) -> PythonType:
    tree = ast.parse(typ_str, mode="eval").body
    return parse_type_from_ast(tree)


def parse_type_expr(annot: cst.BaseExpression, silent=True) -> PythonType | None:
    code = show_expr(annot, quoted=False)
    code = re.sub(r"#.*\n", "", code).replace("\n", "")
    try:
        return parse_type_str(code)
    except Exception as e:
        if silent:
            return None
        else:
            print(f"Failed to parse type expression: `{code}` in source module:")
            raise e


def parse_type_from_ast(tree: ast.expr) -> PythonType:
    assert isinstance(tree, ast.expr)
    match tree:
        case ast.Name() | ast.Attribute():
            return PythonType(parse_qualified_name(tree))
        case ast.Constant(value=str() as s):
            ty = parse_type_from_ast(ast.parse(s, mode="eval").body)
            return ty
        case ast.Constant(value=v):
            if v == None:
                return PythonType(("None",))
            elif v == (...):
                return PythonType(("...",))
            else:
                return PythonType((str(v),))
        case ast.List(elts=elts):  # this can happen inside Callable
            args = tuple(map(parse_type_from_ast, elts))
            return PythonType(("<List>",), args)
        case ast.Subscript(value=(ast.Attribute() | ast.Name()) as v, slice=slice):
            head = parse_qualified_name(v)
            if head[-1] == "Literal":
                return PythonType(head)  # discard the parameters
            match slice:
                case ast.Tuple(elts=elts):
                    args = tuple(map(parse_type_from_ast, elts))
                case _:
                    args = (parse_type_from_ast(slice),)
            return PythonType(head, args)
        case ast.BinOp(left=left, right=right, op=ast.BitOr()):
            return PythonType(
                ("<|>",), (parse_type_from_ast(left), parse_type_from_ast(right))
            )
        case ast.Call():
            return PythonType(("<FuncCall>",))
        case ast.Tuple(elts=elts):
            return PythonType(("<Tuple>",), tuple(map(parse_type_from_ast, elts)))
        case _:
            raise SyntaxError(
                f"Unsupported ast type: {ast.dump(tree, include_attributes=True)}"
            )


def parse_qualified_name(tree: ast.Attribute | ast.Name):
    segs = []
    while isinstance(tree, ast.Attribute):
        segs.append(tree.attr)
        tree = tree.value  # type: ignore
    assert isinstance(tree, ast.Name)
    segs.append(tree.id)
    return tuple(reversed(segs))


class MypyFeedback(NamedTuple):
    position: CodePosition
    message: str
    error_code: str

    def __repr__(self):
        return f"MypyFeedback('[{self.error_code}]{self.position.line}:{self.position.column}: {self.message})'"


class TypeCheckArgs(NamedTuple):
    """
    If in_isolation is True, then each file is treated as a single-file project.
    This can lead to better performance but is less precise.
    """

    no_feedback: bool = False
    check_in_isolation: bool = False

    def __repr__(self):
        return repr_modified_args(self)


@dataclass
class MypyResult:
    # total number of errors in all files
    num_errors: int
    # records the errors in each file and their locations. Absolute paths are recorded.
    error_dict: dict[Path, list[MypyFeedback]]
    # the original output by mypy
    output_str: str


class MypyChecker:

    TypeCheckFlags = [
        "--follow-imports=skip",
        "--namespace-packages",
        "--allow-untyped-globals",
        "--explicit-package-bases",
        "--ignore-missing-imports",  # a hacky workaround
        "--allow-redefinition",
        "--show-column-numbers",
        "--show-error-codes",
        "--soft-error-limit=-1",
        "--no-strict-optional",  # to be compatible with training data
        "--config-file=",
        # "--check-untyped-defs",  # turn off to improve performance
    ]

    Preamble = "from typing import Any, List, Tuple, Dict, Set # SPOT\n"

    MypyErrorsToIgnore = [
        # currently we use a very simple way handle Literal types.
        "Literal[...] must have at least one parameter",
        'The return type of "__init__" must be None',
    ]

    MypyErrorCodesToIgnore = {"valid-type"}

    @staticmethod
    def check_project(proj: Path, binary_path: Path | None = None) -> MypyResult | str:
        if binary_path is None:
            binary_path = proj_root() / ".venv/bin"
        binary_path = binary_path.resolve()
        cmd = [
            binary_path / "python",
            binary_path / "mypy",
            ".",
            *MypyChecker.TypeCheckFlags,
        ]
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=proj,
        )
        return MypyChecker.parse_mypy_output(out, cmd, cwd=proj)

    @staticmethod
    def check_code(
        code: str, cwd: Optional[Path] = None, mypy_path: Path | None = None
    ) -> MypyResult | str:
        "Treat code as a single-file project and performs the type checking."
        if mypy_path is None:
            mypy_path = proj_root() / ".venv/bin/mypy"
        if cwd is None:
            proc = multiprocessing.current_process()
            cwd = proj_root() / "../mypy_temp" / proc.name
        cwd.mkdir(parents=True, exist_ok=True)

        try:
            (cwd / "code.py").write_text(code)
            cmd = [
                "python",
                str(mypy_path),
                "code.py",
                "--check-untyped-defs",
                *MypyChecker.TypeCheckFlags,
            ]
            out = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=cwd,
            )
        finally:
            shutil.rmtree(cwd)
        return MypyChecker.parse_mypy_output(out, cmd, cwd)

    @staticmethod
    def parse_mypy_output(
        output: subprocess.CompletedProcess[str],
        cmd: list,
        cwd: Path,
    ) -> MypyResult | str:
        lines = output.stdout.splitlines()
        if len(lines) == 0:
            return f"mypy failed. Command: `{' '.join(map(str, cmd))}` in cwd='{cwd}'\nError: {output.stderr}"
        error_dict: dict[Path, list[MypyFeedback]] = {}
        for l in lines:
            m = re.match(r"(.*\.py|<string>):(\d+:\d+): error: (.+) \[([a-z\-]+)\]", l)
            if m is not None:
                file = Path(cwd / m.group(1)).resolve()
                line, col = map(int, m.group(2).split(":"))
                msg = m.group(3)
                error_code = m.group(4)
                if error_code in MypyChecker.MypyErrorCodesToIgnore or any(
                    e in msg for e in MypyChecker.MypyErrorsToIgnore
                ):
                    continue
                if file not in error_dict:
                    error_dict[file] = []
                fb = MypyFeedback(CodePosition(line, col), msg, error_code)
                error_dict[file].append(fb)

        # m = re.match(r"Found (\d+) errors? in", lines[-1])
        # if m is None:
        #     num_errors = 0
        # else:
        #     num_errors = int(m.group(1))
        num_errors = sum(map(len, error_dict.values()))

        # total_errors = sum(map(len, error_dict.values()))
        # unfortunately, some errors do not have a column number.
        # assert (
        #     num_errors == total_errors
        # ), f"{num_errors} != {total_errors}. errors found: {error_dict.values()}\n mypy output: \n{output.stdout}"
        return MypyResult(num_errors, error_dict, output.stdout)


class IncrementalChekcer:
    def __init__(self, dmypy_path, code_dir, wait_before_check=1.0) -> None:
        self.code_dir = code_dir
        self.dmypy_path = dmypy_path
        self.wait_before_check: float = wait_before_check

        subprocess.run(
            [
                "python",
                self.dmypy_path,
                "start",
                "--",
                *MypyChecker.TypeCheckFlags,
            ],
            cwd=self.code_dir,
            capture_output=True,
        )
        sout = subprocess.run(
            ["python", self.dmypy_path, "check", "."],
            cwd=self.code_dir,
            capture_output=True,
            text=True,
        )
        if sout.stderr:
            logging.warn(f"Mypy failed on initial check: {sout.stderr}")

    def close(self):
        subprocess.run(
            ["python", self.dmypy_path, "stop"],
            cwd=self.code_dir,
            capture_output=True,
        )

    def recheck_files(self, *updated_files: Path) -> MypyResult | str:
        # TODO: remove this workaround once (https://github.com/python/mypy/issues/12697) is fixed.
        time.sleep(
            self.wait_before_check
        )  # wait to make sure the type checker sees the file changes

        out = self._run_mypy(
            [
                "python",
                self.dmypy_path,
                "recheck",
                # "--perf-stats-file",
                # "mypy_perf.json",
                "--update",
                *map(str, updated_files),
            ]
        )

        return out

    def recheck_project(self) -> MypyResult:
        # TODO: remove this workaround once (https://github.com/python/mypy/issues/12697) is fixed.
        time.sleep(
            self.wait_before_check
        )  # wait to make sure the type checker sees the file changes

        out = self._run_mypy(
            [
                "python",
                self.dmypy_path,
                "check",
                ".",
            ]
        )
        assert isinstance(out, MypyResult), f"Recheck failed: {out}"
        return out

    def _run_mypy(self, cmd: list[str]) -> MypyResult | str:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=self.code_dir,
        )

        return MypyChecker.parse_mypy_output(result, cmd, cwd=Path(self.code_dir))


@contextmanager
def mypy_checker(code_dir: Path, dmypy_path: Path | None = None, wait_before_check=1.0):
    checker = None
    try:
        if dmypy_path is None:
            dmypy_path = proj_root() / ".venv/bin/dmypy"
        yield (
            checker := IncrementalChekcer(
                dmypy_path, code_dir, wait_before_check=wait_before_check
            )
        )
    finally:
        if checker is not None:
            checker.close()


def count_type_frequency(
    types: Iterable[PythonType], recursive: bool = True
) -> Counter[str]:
    counter = Counter[str]()

    def count_type(t: PythonType):
        counter[t.head_name()] += 1
        if recursive:
            for c in t.args:
                count_type(c)

    for t in types:
        count_type(t)

    return counter
