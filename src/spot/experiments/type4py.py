import requests
from spot.function_dataset import SignatureMap
from spot.type_check import normalize_type, parse_type_expr
from spot.utils import *
from spot.static_analysis import (
    ProjectPath,
    ModuleName,
    ElemSignature,
    PythonProject,
    VariableSignature,
    FunctionSignature,
)

PredList = list[list]  # of the form [[type1, score1], [type2, score2], ...]


class Type4PyResponseParser:
    def __init__(self, module: ModuleName):
        self.module = module
        self.assignment: dict[ProjectPath, ElemSignature] = dict()

    def parse(self, res_json: dict) -> dict[ProjectPath, ElemSignature]:
        self.assignment = dict()
        res = res_json["response"]

        for name, pred in res["variables_p"].items():
            annot = self.parse_prediction(pred)
            sig = VariableSignature(annot, in_class=False)
            self.assignment[ProjectPath(self.module, name)] = sig
        for f in res["funcs"]:
            self._parse_func(
                f,
                ProjectPath(self.module, ""),
                in_class=False,
            )
        for c in res["classes"]:
            self._parse_cls(c, ProjectPath(self.module, ""))
        return self.assignment

    def _parse_cls(self, cls_json: dict, base: ProjectPath):
        attr_preds: dict[str, PredList] = cls_json["variables_p"]
        new_base = base.append(cls_json["name"])
        for name, pred in attr_preds.items():
            annot = self.parse_prediction(pred)
            sig = VariableSignature(annot, in_class=True)
            self.assignment[new_base.append(name)] = sig
        for func_json in cls_json["funcs"]:
            try:
                self._parse_func(func_json, new_base, in_class=True)
            except:
                print(f"Failed to parse function")
                print("JSON:")
                display(func_json)
                raise

    @staticmethod
    def parse_prediction(pred: PredList) -> cst.Annotation | None:
        if pred:
            return cst.Annotation(cst.parse_expression(pred[0][0]))
        else:
            return None

    def _parse_func(self, func_json: dict, base: ProjectPath, in_class: bool):
        preds = func_json["params_p"]
        params_pred = {
            v: self.parse_prediction(preds[v]) for v in preds if len(preds[v]) > 0
        }
        ret_pred = None
        if "ret_type_p" in func_json:
            ret_pred = self.parse_prediction(func_json["ret_type_p"])
        if ret_pred is None:
            ret_pred = cst.Annotation(cst.parse_expression("None"))
        sig = FunctionSignature(
            params_pred,
            ret_pred,
            in_class=in_class,
        )
        self.assignment[base.append(func_json["name"])] = sig


def run_type4py_request(
    code: str, module: ModuleName
) -> dict[ProjectPath, ElemSignature] | str:
    res = requests.post("https://type4py.com/api/predict?tc=0", code).json()
    if res["response"] is None:
        return res["error"]
    return Type4PyResponseParser(module).parse(res)


_DefaultImport = cst.parse_statement(
    "from typing import Any, List, Tuple, Dict, Set, Union, Optional"
)


def remove_newer_syntax(m: cst.Module) -> cst.Module:
    """
    Remove or rewrite any newer python features that Type4Py doesn't support.
    """

    class PatternRewriter(cst.CSTTransformer):
        def leave_MatchAs(self, node, updated: cst.MatchAs):
            if updated.pattern:
                return updated.pattern
            elif updated.name:
                return updated.name
            else:
                # wild card pattern
                return cst.Name("_")

    def pattern_to_expr(pattern: cst.MatchPattern):
        np = cast(cst.BaseExpression, pattern.visit(PatternRewriter()))
        return cst.parse_expression(m.code_for_node(np))

    class Rewriter(cst.CSTTransformer):
        def leave_Annotation(self, node, updated: "cst.Annotation"):
            ty = parse_type_expr(m, updated.annotation, silent=True)
            if ty is None:
                return cst.RemoveFromParent()
            ty = normalize_type(ty)  # this should get rid of the Union type syntax.
            return updated.with_changes(annotation=cst.parse_expression(str(ty)))

        def leave_Module(self, node, updated: "cst.Module"):
            new_lines = [_DefaultImport]
            new_lines.extend(updated.body)
            return updated.with_changes(body=new_lines)

        def leave_Match(self, node, updated: cst.Match):
            subject = updated.subject
            if isinstance(subject, cst.Tuple):
                subject = subject.with_changes(
                    lpar=[cst.LeftParen()], rpar=[cst.RightParen()]
                )

            conditions = [
                cst.Comparison(
                    subject,
                    [
                        cst.ComparisonTarget(
                            cst.Equal(),
                            pattern_to_expr(c.pattern),
                        )
                    ],
                )
                for c in updated.cases
            ]
            bodies = [c.body for c in updated.cases]
            if_clauses = None
            for cond, body in reversed(list(zip(conditions, bodies))):
                if_clauses = cst.If(cond, body, orelse=if_clauses)
            assert isinstance(if_clauses, cst.If)
            return if_clauses

    return m.visit(Rewriter())


@dataclass
class Type4PyEvalResult:
    pred_maps: dict[str, SignatureMap]
    label_maps: dict[str, SignatureMap]


def eval_type4py_on_project(
    project: PythonProject,
    max_workers: int = 4,
) -> Type4PyEvalResult:
    module_srcs = {
        name: remove_newer_syntax(m.tree).code for name, m in project.modules.items()
    }
    model_outputs = pmap(
        run_type4py_request,
        list(module_srcs.values()),
        list(module_srcs.keys()),
        desc="Calling Type4Py",
        max_workers=max_workers,
    )

    label_signatures = {e.path: e.get_signature() for e in project.all_elems()}

    pred_signatures = dict[ProjectPath, ElemSignature]()
    for m, o in zip(module_srcs.keys(), model_outputs):
        if isinstance(o, str):
            if list(project.modules[m].all_elements()):
                # only warn for non-empty modules
                logging.warning(f"In module {m}, Type4Py errored: {o}")
        else:
            pred_signatures.update(o)

    proj_name = project.root_dir.name

    return Type4PyEvalResult(
        pred_maps={proj_name: pred_signatures},
        label_maps={proj_name: label_signatures},
    )
