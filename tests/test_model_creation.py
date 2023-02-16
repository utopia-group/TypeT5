import torch
from transformers import AutoConfig
from transformers.models.t5.configuration_t5 import T5Config

from typet5.data import CtxArgs
from typet5.model import DecodingArgs, ModelType, ModelWrapper
from typet5.train import TrainingConfig
from typet5.utils import DefaultTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_basic_torch_operations():
    x = torch.randn(10).to(device)
    assert x.sum() <= 10


def test_model_creation():
    config = AutoConfig.from_pretrained(ModelWrapper.get_codet5_path())
    model = ModelType(config).to(device)

    ctx_args = TrainingConfig().dec_ctx_args()
    dec_args = DecodingArgs(ctx_args, 32)
    wrapper = ModelWrapper(model, DefaultTokenizer, dec_args, set())
    wrapper.to(device)

    ids = DefaultTokenizer.encode(
        "def get_count() -> <extra_id_0>: ...", return_tensors="pt"
    )
    batch = {"input_ids": ids, "n_labels": [1]}
    out = wrapper.predict_on_batch(batch)
    assert True
