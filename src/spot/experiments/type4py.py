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
        self.assignment = dict[ProjectPath, ElemSignature]()

    def parse(self, res_json: dict):
        res = res_json["response"]
        return res

    def parse_cls(self, cls_json: dict, base: ProjectPath):
        attr_preds: dict[str, PredList] = cls_json["variables_p"]
        for name, pred in attr_preds.items():
            annot = cst.Annotation(cst.parse_expression(pred[0][0]))
            sig = VariableSignature(annot, in_class=True)
            self.assignment[base.append(name)] = sig
