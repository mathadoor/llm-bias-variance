from unittest.mock import MagicMock, patch

import pytest
from transformers import GenerationMixin, PreTrainedTokenizerBase

from shared.hf_utils import HFGenerator


def _get_model_config():
    return {
        "name": "test",
        "device": "cpu",
        "model": {},
        "tokenizer": {"padding_side": "left"},
    }


class TestHFGeneratorInit:
    @pytest.fixture
    def mocks(self):
        model = MagicMock(spec=GenerationMixin)
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        with patch("transformers.AutoModel.from_pretrained", return_value=model) as m_model, \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as m_tok:
            yield m_model, m_tok, model, tokenizer

    def test_init_loads_model_and_tokenizer(self, mocks):
        # Arrange
        m_model, m_tok, _, _ = mocks
        config = _get_model_config()

        # Act
        HFGenerator(config)

        # Assert
        m_model.assert_called_once_with(config["name"], **config["model"])
        m_tok.assert_called_once_with(config["name"], **config["tokenizer"])

    def test_init_model_supports_generation(self, mocks):
        # Arrange
        _, _, model, tokenizer = mocks

        # Act
        generator = HFGenerator(_get_model_config())

        # Assert
        assert generator.model is model
        assert generator.tokenizer is tokenizer
        assert isinstance(generator.model, GenerationMixin)
        assert hasattr(generator.model, "generate")

    def test_init_non_generative_model_raises_error(self):
        # Arrange
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        with patch("transformers.AutoModel.from_pretrained", return_value=MagicMock(spec=object)), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer):
            # Act / Assert
            with pytest.raises(TypeError):
                HFGenerator(_get_model_config())

    def test_init_invalid_tokenizer_raises_error(self):
        # Arrange
        model = MagicMock(spec=GenerationMixin)
        with patch("transformers.AutoModel.from_pretrained", return_value=model), \
             patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock(spec=object)):
            # Act / Assert
            with pytest.raises(TypeError):
                HFGenerator(_get_model_config())
