from collections import Counter

import numpy as np
from transformers import RobertaTokenizer
from transformers.models.t5 import T5ForConditionalGeneration

from spot.utils import *

TokenizerSPOT = RobertaTokenizer
ModelSPOT = T5ForConditionalGeneration


class AugModel:
    """A type inference model augmented with the feedback from the type checker"""

    def __init__(self, model, tokenizer):
        self.model: ModelSPOT = model
        self.tokenizer: TokenizerSPOT = tokenizer

    def predict(self, input_text: str, mask_type_annots: bool = True):
        pass
