# files containing utils for computing model input contexts

from .utils import *


@dataclass
class DynCtxArgs:
    # the max number of code tokens to the left of the prediction
    left_ctx: int = 512
    # the max number of code tokens to the left of the prediction
    right_ctx: int = 512
    # the max additional context containing information such as imports and stubs
    extra_ctx: int = 512
    include_imports: bool = True


def extract_inputs_dyn_ctx(src: TokenizedSrc, args: DynCtxArgs):
    pass
