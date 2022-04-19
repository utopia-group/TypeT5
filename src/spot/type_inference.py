from time import sleep
from .type_checking import *

TypeExpr = cst.BaseExpression

@dataclass
class TypeInfState:
    """The current (partically annotated) CST"""
    module: cst.Module
    to_annot: list[AnnotPath]
    annotated: dict[AnnotPath, TypeExpr]
    num_errors: int

    def __str__(self):
        return f"""
num_errors: {self.num_errors}
num_to_annot: {len(self.to_annot)}
to_annotate: {self.to_annot}
------------------------ code -------------------------------
{self.module.code}
        """


@dataclass
class TypeInfAction:
    """Annotate a location with a type. The type will be converted to `Any` if it would trigger a type error."""
    path: AnnotPath
    type: TypeExpr

class TypeInfEnv:
    """An [OpenAI Gym](https://gym.openai.com/docs/)-style environment for 
    sequentially annotating a python source file.
    
    """
    def __init__(self, checker: MypyChecker, src_file: str):
        self.checker = checker
        self.src_file = realpath(src_file)
        self.original_src = read_file(src_file)
        self.state = TypeInfState(cst.Module([]), [], {}, 0)
        self.reset()

    def reset(self) -> None:
        write_file(self.src_file, self.original_src)
        self.state.module = cst.parse_module(self.original_src)
        paths, annots = collect_annotations(self.state.module)
        self.state.annotated = {k: v.annotation for k, v in annots.items()}
        self.state.to_annot = [p for p in paths if p not in annots]
        self.state.num_errors = self.checker.recheck_files(self.src_file).num_errors

    def step(self, action: TypeInfAction, check_any=False) -> None:
        assert action.path in self.state.to_annot, f"Invalid action: path {action.path} already annotated."
        type = action.type
        mod = apply_annotations(self.state.module, {action.path: cst.Annotation(type)})
        write_file(self.src_file, mod.code)
        ne = self.checker.recheck_files(self.src_file).num_errors
        if ne > self.state.num_errors:
            type = cst.Name("Any")
            mod = apply_annotations(self.state.module, {action.path: cst.Annotation(type)})
            write_file(self.src_file, mod.code)
            if check_any:
                check_r = self.checker.recheck_files(self.src_file)
                assert check_r.num_errors == self.state.num_errors, (
                    "Adding Any should not trigger more type errors.\n"
                    f"action: {action}\n"
                    f"mypy output: {check_r.output_str}\n"
                    f"---------code---------\n {mod.code}\n"
                )
        self.state.to_annot.remove(action.path)
        self.state.annotated[action.path] = type
        self.state.module = mod
    