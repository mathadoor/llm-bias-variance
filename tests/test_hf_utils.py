from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import BatchEncoding, GenerationMixin, PreTrainedTokenizerBase

from shared.hf_utils import HFGenerator, HFLogitProcessor


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


CHAT = [{"role": "user", "content": "hi"}]


def _make_generator(generate_output):
    """Build an HFGenerator whose tokenizer/model are mocks returning `generate_output`."""
    model = MagicMock(spec=_CausalLM)
    model.generate.return_value = generate_output
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.apply_chat_template.return_value = BatchEncoding({"input_ids": torch.tensor([[1, 2, 3]])})
    tokenizer.decode.return_value = "decoded"
    with _patch_loaders(model, tokenizer):
        return HFGenerator(_get_model_config()), model, tokenizer


def _output(**kwargs):
    return SimpleNamespace(**kwargs)


class TestGenerateChat:
    def test_returns_text_tokens_and_logits(self):
        # Arrange
        seqs, logits = torch.tensor([4, 5, 6]), (torch.rand(1, 2),)
        gen, _, tok = _make_generator(_output(sequences=seqs, logits=logits))

        # Act
        result = gen.generate_chat(CHAT, {})

        # Assert
        assert result["output_text"] == "decoded"
        assert torch.equal(result["output_tokens"], seqs)
        assert result["logits"] is logits
        assert torch.equal(result["input_tokens"]["input_ids"], torch.tensor([[1, 2, 3]]))
        tok.decode.assert_called_once()
        assert tok.decode.call_args.kwargs["skip_special_tokens"] is True

    def test_logits_omitted_when_not_returned(self):
        gen, _, _ = _make_generator(_output(sequences=torch.tensor([4])))

        result = gen.generate_chat(CHAT, {})

        assert "logits" not in result

    def test_missing_sequences_raises_error(self):
        gen, _, _ = _make_generator(_output())

        with pytest.raises(ValueError):
            gen.generate_chat(CHAT, {})

    def test_generation_config_forwarded_with_dict_output(self):
        gen, model, _ = _make_generator(_output(sequences=torch.tensor([4])))

        gen.generate_chat(CHAT, {"max_new_tokens": 5})

        kwargs = model.generate.call_args.kwargs
        assert kwargs["max_new_tokens"] == 5
        assert kwargs["return_dict_in_generate"] is True

    def test_caller_generation_config_not_mutated(self):
        gen, _, _ = _make_generator(_output(sequences=torch.tensor([4])))
        config = {"max_new_tokens": 5}

        gen.generate_chat(CHAT, config)

        assert config == {"max_new_tokens": 5}

    @pytest.mark.parametrize("chat", [CHAT[0], CHAT])
    def test_accepts_single_message_or_list(self, chat):
        gen, _, tok = _make_generator(_output(sequences=torch.tensor([4])))

        gen.generate_chat(chat, {})

        assert tok.apply_chat_template.call_args.args[0] is chat

class TestLogitProcessorCall:

    @pytest.mark.parametrize(
        "input_ids, scores, result", 
        [
            ([[1, 2], [2, 3]], [[0.2, 0.4, 0.8], [0.3, 0.5, 0.2]], [[0.2, 0.4], [0.3, 0.5]]),
            ([1, 2, 4], [0.2, 0.4, 0.8], [[0.2, 0.4]]),
            ([[1], [2]], [[0.2, 0.4, 0.8], [0.3, 0.5, 0.2]], [[0.2, 0.4], [0.3, 0.5]]),
            ([1], [0.2, 0.4, 0.8], [[0.2, 0.4]])
        ]
    )
    def test_logits_correctness(self, input_ids, scores, result):
        logit_processor = HFLogitProcessor(label_index=[0, 1]) 
        logit_processor(
            torch.tensor(input_ids),
            torch.tensor(scores)
        )
        stored_scores = logit_processor.get_scores()
        assert torch.equal(
            stored_scores,
            torch.tensor(result)
        )

    # logits accumulate and clear upon reset
    @pytest.mark.parametrize(
        "input_id_1, scores_1, input_id_2, scores_2, result",
        [
            ([1, 2], [0.2, 0.4, 0.5], [1, 2, 3], [0.4, 0.8, 0.9], [[0.2, 0.4], [0.4, 0.8]]),
            ([1], [0.2, 0.4, 0.5], [1, 3], [0.4, 0.8, 0.9], [[0.2, 0.4], [0.4, 0.8]])
        ]
    )
    def test_logits_multiple_calls_accumulate_and_clears_upon_reset(
        self, input_id_1, scores_1, input_id_2, scores_2, result
    ):
        logit_processor = HFLogitProcessor(label_index=[0, 1]) 
        for input_ids, scores in [(input_id_1, scores_1), (input_id_2, scores_2)]:
            logit_processor(
                torch.tensor(input_ids),
                torch.tensor(scores)
            )
        stored_scores = logit_processor.get_scores()
        logit_processor.reset()
        post_reset_score = logit_processor.get_scores()

        assert torch.equal(
            stored_scores,
            torch.tensors(result)
        )
        assert torch.equal(
            post_reset_score, 
            torch.tensor([])
        )