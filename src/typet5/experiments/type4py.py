import requests

from typet5.experiments.utils import SupportedSyntax, remove_newer_syntax
from typet5.function_dataset import SignatureMap
from typet5.static_analysis import (
    ElemSignature,
    FunctionSignature,
    ModuleName,
    ProjectPath,
    PythonProject,
    VariableSignature,
    reorder_signature_map,
)
from typet5.type_check import normalize_type, parse_type_expr
from typet5.utils import *

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
    res = requests.post("https://type4py.com/api/predict?tc=0", code.encode()).json()
    if res["response"] is None:
        return res["error"]
    return Type4PyResponseParser(module).parse(res)


@dataclass
class Type4PyEvalResult:
    pred_maps: dict[str, SignatureMap]
    label_maps: dict[str, SignatureMap]

    def __post_init__(self):
        # reorder the function args to match the labels
        for pname, pred_map in self.pred_maps.items():
            if pname not in self.label_maps:
                continue
            label_map = self.label_maps[pname]
            self.pred_maps[pname] = reorder_signature_map(pred_map, label_map)


Type4PySupportedSyntax = SupportedSyntax(
    pattern_match=False, union_types=False, basic_types=False
)


def eval_type4py_on_projects(
    projects: list[PythonProject],
    max_workers: int = 4,
) -> Type4PyEvalResult:
    name2project = {p.name: p for p in projects}
    module_srcs = {
        (project.name, name): remove_newer_syntax(m.tree, Type4PySupportedSyntax).code
        for project in projects
        for name, m in project.modules.items()
    }
    model_outputs = pmap(
        run_type4py_request,
        list(module_srcs.values()),
        [mname for pname, mname in module_srcs.keys()],
        desc="Calling Type4Py",
        max_workers=max_workers,
    )

    label_signatures: dict[str, SignatureMap] = {
        project.name: {e.path: e.get_signature() for e in project.all_elems()}
        for project in projects
    }

    pred_signatures: dict[str, SignatureMap] = {n: dict() for n in label_signatures}
    for (pname, mname), o in zip(module_srcs.keys(), model_outputs):
        if isinstance(o, str):
            if list(name2project[pname].modules[mname].all_elements()):
                # only warn for non-empty modules
                logging.warning(
                    f"In project {pname} module {mname}, Type4Py errored: {o}"
                )
        else:
            pred_signatures[pname].update(o)

    return Type4PyEvalResult(
        pred_maps=pred_signatures,
        label_maps=label_signatures,
    )
