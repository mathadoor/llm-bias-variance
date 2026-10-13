from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import GenerationMixin, PreTrainedTokenizerBase

from shared.hf_utils import HFGenerator


def _get_model_config():
    return {
        "name": "test",
        "device": "cpu",
        "model": {},
        "tokenizer": {"padding_side": "left"},
    }


class _CausalLM(torch.nn.Module, GenerationMixin):
    """Spec for mock models: supports both .to(device) and .generate()."""


@contextmanager
def _patch_loaders(model=None, tokenizer=None):
    model = MagicMock(spec=_CausalLM) if model is None else model
    model.to.return_value = model 
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase) if tokenizer is None else tokenizer
    with patch("shared.hf_utils.AutoModelForCausalLM.from_pretrained", return_value=model) as m_model, \
         patch("shared.hf_utils.AutoTokenizer.from_pretrained", return_value=tokenizer) as m_tok:
        yield m_model, m_tok


class TestHFGeneratorInit:
    def test_init_loads_model_and_tokenizer(self):
        # Arrange
        config = _get_model_config()

        # Act
        with _patch_loaders() as (m_model, m_tok):
            HFGenerator(config)

        # Assert
        m_model.assert_called_once_with(config["name"], **config["model"])
        m_tok.assert_called_once_with(config["name"], **config["tokenizer"])

    def test_init_model_supports_generation(self):
        # Arrange
        model = MagicMock(spec=_CausalLM)
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

        # Act
        with _patch_loaders(model, tokenizer):
            generator = HFGenerator(_get_model_config())

        # Assert
        assert generator.model is model
        assert generator.tokenizer is tokenizer
        assert isinstance(generator.model, GenerationMixin)
        assert hasattr(generator.model, "generate")

    def test_init_moves_model_to_device(self):
        # Arrange
        model = MagicMock(spec=_CausalLM)
        config = _get_model_config()

        # Act
        with _patch_loaders(model):
            HFGenerator(config)

        # Assert
        model.to.assert_called_once_with(config["device"])

    def test_init_non_generative_model_raises_error(self):
        with _patch_loaders(model=MagicMock(spec=torch.nn.Module)), pytest.raises(TypeError):
            HFGenerator(_get_model_config())

    def test_init_invalid_tokenizer_raises_error(self):
        with _patch_loaders(tokenizer=MagicMock(spec=object)), pytest.raises(TypeError):
            HFGenerator(_get_model_config())
