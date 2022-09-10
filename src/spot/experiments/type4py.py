from spot.utils import *
from spot.static_analysis import (
    ProjectPath,
    ModuleName,
    ElemSignature,
    VariableSignature,
    FunctionSignature,
)
from spot.type_env import parse_type_str

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
        if "ret_type_p" in func_json:
            ret_pred = self.parse_prediction(func_json["ret_type_p"])
        else:
            ret_pred = None
        sig = FunctionSignature(
            params_pred,
            ret_pred,
            in_class=in_class,
        )
        self.assignment[base.append(func_json["name"])] = sig
