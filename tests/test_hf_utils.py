from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from datasets import Dataset, DatasetDict
from transformers import BatchEncoding, GenerationMixin, PreTrainedTokenizerBase, AutoTokenizer, AutoModelForCausalLM

from shared.hf_utils import HFGenerator, HFLogitProcessor, load_hf_dataset


def _get_model_config():
    return {
        "name": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "device": "cpu",
        "model": {},
        "tokenizer": {"padding_side": "left"},
    }

def _get_generation_config(
    store_probs: bool = True,
    labels: list[str] = ["0", "1"] 
):
    return dict(
        store_probs=store_probs,
        labels=labels,
        temperature=0.0,
        do_sample=False,
        max_new_tokens=1000,
        num_beams=1
    ) 
    
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
    def test_init_loads_model_and_tokenizer_and_label_indices(self):
        # Arrange
        config = _get_model_config()
        gen_config = _get_generation_config()

        # Act
        with _patch_loaders() as (m_model, m_tok):
            generator = HFGenerator(config, gen_config)

        # Assert
        m_model.assert_called_once_with(config["name"], **config["model"])
        m_tok.assert_called_once_with(config["name"], **config["tokenizer"])
        store_probs = gen_config.get("store_probs")
        labels = gen_config.get("labels", [])

        if len(labels) > 0 and store_probs:
            assert generator.tokenizer.call_args.args == (labels, )

    def test_init_model_supports_generation(self):
        # Arrange
        model = MagicMock(spec=_CausalLM)
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)

        # Act
        with _patch_loaders(model, tokenizer):
            generator = HFGenerator(_get_model_config(), _get_generation_config())

        # Assert
        assert generator.model is model
        assert generator.tokenizer is tokenizer
        assert isinstance(generator.model, GenerationMixin)
        assert hasattr(generator.model, "generate")

    @pytest.mark.parametrize(
            "store_probs, labels",
            [(True, []), (True, ["0", "1"])]
    )
    def test_init_model_updates_generation_config(self, store_probs, labels):
        # Arrange
        model = MagicMock(spec=_CausalLM)
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        gen_config = _get_generation_config(store_probs=store_probs, labels=labels)

        tokenizer.return_value = BatchEncoding({
            'input_ids': [[0], [1]],
            'attention_mask': torch.flip(torch.triu(torch.ones(2, 2)), dims=[1])
        }) 
    
        # Act
        with _patch_loaders(model, tokenizer):
            generator = HFGenerator(_get_model_config(), gen_config)


        updated_gen_config = gen_config.copy()
        del updated_gen_config["store_probs"]
        del updated_gen_config["labels"]

        flag = store_probs and len(labels) == 0
        updated_gen_config["output_logits"] = flag 
        updated_gen_config["return_dict_in_generate"] = True
        
        labels = [] if flag else [0, 1]

        # Assert
        assert generator.label_index == labels
        assert generator.generation_config == updated_gen_config


    @pytest.mark.parametrize("input_ids", [[[0, 1], [2]], [[0], []]])
    def test_init_label_not_single_token_raises_error(self, input_ids):
        tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
        tokenizer.return_value = BatchEncoding({"input_ids": input_ids})

        with _patch_loaders(tokenizer=tokenizer), pytest.raises(ValueError):
            HFGenerator(_get_model_config(), _get_generation_config())

    def test_init_moves_model_to_device(self):
        # Arrange
        model = MagicMock(spec=_CausalLM)
        config = _get_model_config()
        gen_config = _get_generation_config()

        # Act
        with _patch_loaders(model):
            HFGenerator(config, gen_config)

        # Assert
        model.to.assert_called_once_with(config["device"])

    def test_init_non_generative_model_raises_error(self):
        with _patch_loaders(model=MagicMock(spec=torch.nn.Module)), pytest.raises(TypeError):
            HFGenerator(_get_model_config(), _get_generation_config())

    def test_init_invalid_tokenizer_raises_error(self):
        with _patch_loaders(tokenizer=MagicMock(spec=object)), pytest.raises(TypeError):
            HFGenerator(_get_model_config(), _get_generation_config())


CHAT = [{"role": "user", "content": "hi"}]


def _make_generator(generate_output, output=["decoded"], input_ids=torch.tensor([[1, 2, 3]])):
    """Build an HFGenerator whose tokenizer/model are mocks returning `generate_output`."""
    model = MagicMock(spec=_CausalLM)
    model.generate.return_value = generate_output
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.apply_chat_template.return_value = BatchEncoding({"input_ids": input_ids})
    tokenizer.batch_decode.return_value = output
    with _patch_loaders(model, tokenizer):
        return HFGenerator(_get_model_config(), _get_generation_config()), model, tokenizer


def _output(**kwargs):
    return SimpleNamespace(**kwargs)


class TestGenerateChat:
    def test_returns_text_tokens_and_logits(self):
        # Arrange
        # prompt is [1, 2, 3]; sequences = prompt + 3 generated tokens
        seqs, logits = torch.tensor([[1, 2, 3, 4, 5, 6]]), (torch.rand(1, 2), )
        gen, _, tok = _make_generator(_output(sequences=seqs, logits=logits))
        expected_logits = torch.softmax(logits[0], dim=-1).unsqueeze(0)

        # Act
        result = gen.generate_chat(CHAT)

        # Assert
        assert result["output_text"] == ["decoded"]
        assert torch.equal(result["output_tokens"], seqs[:, 3:])
        assert torch.equal(result["logits"], expected_logits)
        assert torch.equal(result["input_tokens"]["input_ids"], torch.tensor([[1, 2, 3]]))
        tok.batch_decode.assert_called_once()
        # only the generated part is decoded, not the prompt
        assert torch.equal(tok.batch_decode.call_args.args[0], seqs[:, 3:])
        assert tok.batch_decode.call_args.kwargs["skip_special_tokens"] is True

    @pytest.mark.parametrize(
        "store_probs, labels, o_probs_present, l_output_probs",
        [
            (True, ["0", "1"], True, 2), 
            (True, [], True, -1), # -1 means vocab size length 
            (False, ["0", "1"], False, None), 
            (False, [], False, None)
        ]
    )
    def test_chat_returns_output_probs_length(
        self,
        store_probs, labels, o_probs_present, l_output_probs
    ):
         # Arrange
        model_config, gen_config = _get_model_config(), _get_generation_config(store_probs, labels)
        gen = HFGenerator(model_config, gen_config)
        expected_length = gen.tokenizer.vocab_size if l_output_probs == -1 else l_output_probs

        # Act
        result = gen.generate_chat(CHAT)

        # Assert
        if o_probs_present:
            assert result.get("logits").size()[-1] == expected_length
        else:
            assert result.get("logits") is None
        assert ("logits" in result) == o_probs_present


    def test_logits_omitted_when_not_returned(self):
        gen, _, _ = _make_generator(_output(sequences=torch.tensor([[1, 2, 3, 4]])))

        result = gen.generate_chat(CHAT)

        assert "logits" not in result

    def test_missing_sequences_raises_error(self):
        gen, _, _ = _make_generator(_output())

        with pytest.raises(ValueError):
            gen.generate_chat(CHAT)

        
    @pytest.mark.parametrize("chat", [CHAT[0], CHAT])
    def test_accepts_single_message_or_list(self, chat):
        gen, _, tok = _make_generator(_output(sequences=torch.tensor([[1, 2, 3, 4]])))

        gen.generate_chat(chat)

        assert tok.apply_chat_template.call_args.args[0] is chat


def test_get_label_logits_returns_correct_probs():
    # Arrange
    pieces = ["positive sentiment.", " 0 is", " the label", ". ###", " output", " 1", " hello"]
    output_text = ["".join(pieces)]          
    output_tokens = torch.arange(4, 11).unsqueeze(0)   
    gen, _, tokenizer = _make_generator(_output(sequences=torch.zeros(1, 3)))  

    tokenizer.decode.side_effect = lambda ids, **kw: "".join(pieces[: len(ids)])
    output_dict = gen.generate_chat("chat")

    torch.manual_seed(0)
    logits = torch.softmax(torch.rand(1, 7, 2), dim=-1)  
    output_dict = {"output_text": output_text, "output_tokens": output_tokens, "logits": logits}

    # Act
    result = gen.get_label_logits("### output", output_dict)

    # Assert: header ends after " output" (step 4), so the answer is step 5
    assert torch.equal(result, logits[:, 5, :])



@pytest.mark.parametrize(
    "pieces, label_index, expected",
    [
        # (header used, position in labels, step of the answer token) or None
        (["positive sentiment.", " 0 is", " the label", ". ###", " output", " 1", " hello"], [8, 9], ("### output", 1, 5)),
        (["reasoning ", "</think>", "\n\n", "1", " done"], [6, 7], ("</think>", 1, 3)),   # skips whitespace
        (["### output", " [label]", " </think>", " 0"], [7], ("</think>", 0, 3)),          # primary not followed by a label
        (["no header here ", "0"], [5], None),
        (["</think>", " ok", " 0"], [6], None),                                           # next token is not a label
        (["answer </think>"], [4], None),                                                 # nothing after the header
    ],
)
def test_get_answer_probs(pieces, label_index, expected):
    gen, _, tokenizer = _make_generator(_output(sequences=torch.zeros(1, 3)))
    tokenizer.decode.side_effect = lambda ids, **kw: "".join(pieces[: len(ids)])
    gen.label_index = label_index
    logits = torch.softmax(torch.rand(1, len(pieces), 2), dim=-1)
    output_dict = {"output_tokens": torch.arange(4, 4 + len(pieces)).unsqueeze(0), "logits": logits}

    answer = gen.get_answer_probs(["### output", "</think>"], output_dict)[0]

    if expected is None:
        assert answer is None
    else:
        header, label, step = expected
        assert answer[:2] == (header, label)
        assert torch.equal(answer[2], logits[0, step])


class TestLogitProcessorCall:

    @pytest.mark.parametrize(
        "input_ids, scores, label_index, result",
        [
            ([[1, 2], [2, 3]], [[0.2, 0.4, 0.8], [0.3, 0.5, 0.2]], [0, 1], [[[0.247309, 0.302064], [0.319873, 0.390694]]]),
            ([[1, 2, 4]], [[0.2, 0.4, 0.8]], [0, 1], [[[0.247309, 0.302064]]]),
            ([[1], [2]], [[0.2, 0.4, 0.8], [0.3, 0.5, 0.2]], [0, 1], [[[0.247309, 0.302064], [0.319873, 0.390694]]]),
            ([[1]], [[0.2, 0.4, 0.8]], [0, 1], [[[0.247309, 0.302064]]]),
            ([[1, 2], [2, 3]], [[0.2, 0.4, 0.8], [0.3, 0.5, 0.2]], [1, 0], [[[0.302064, 0.247309], [0.390694, 0.319873]]]),
            ([[1]], [[0.2, 0.4, 0.8]], [1, 0], [[[0.302064, 0.247309]]]),
        ]
    )
    def test_probs_correctness(self, input_ids, scores, label_index, result):
        logit_processor = HFLogitProcessor(label_index=label_index)
        returned_score = logit_processor(
            torch.tensor(input_ids),
            torch.tensor(scores)
        )
        stored_scores = logit_processor.get_probs()
        # expected values are written (steps, batch, labels); get_probs is (batch, steps, labels)
        assert torch.allclose(
            stored_scores,
            torch.tensor(result).transpose(0, 1),
            atol=1e-6
        )
        # the processor must pass the raw scores through untouched
        assert torch.equal(
            returned_score,
            torch.tensor(scores)
        )

    # logits accumulate and clear upon reset
    @pytest.mark.parametrize(
        "input_id_1, scores_1, input_id_2, scores_2, label_index, result",
        [
            ([[1, 2]], [[0.2, 0.4, 0.5]], [[1, 2, 3]], [[0.4, 0.8, 0.9]], [0, 1], [[[0.280013, 0.342009]], [[0.241514, 0.360297]]]),
            ([[1]], [[0.2, 0.4, 0.5]], [[1, 3]], [[0.4, 0.8, 0.9]], [0, 1], [[[0.280013, 0.342009]], [[0.241514, 0.360297]]]),
            (
                [[1, 2], [2, 3]], 
                [[0.2, 0.4, 0.8], [0.3, 0.5, 0.2]], 
                [[1, 2, 3], [2, 3, 2]], 
                [[0.2, 0.9, 0.3], [0.9, 0.5, 0.2]], 
                [0, 1],
                [[[0.247309, 0.302064], [0.319873, 0.390694]], [[0.242782, 0.488903], [0.461488, 0.309344]]]
            ),
            # switched order: every step follows label_index order
            ([[1, 2]], [[0.2, 0.4, 0.5]], [[1, 2, 3]], [[0.4, 0.8, 0.9]], [1, 0], [[[0.342009, 0.280013]], [[0.360297, 0.241514]]]),
            (
                [[1, 2], [2, 3]], 
                [[0.2, 0.4, 0.8], [0.3, 0.5, 0.2]], 
                [[1, 2, 3], [2, 3, 2]], 
                [[0.2, 0.9, 0.3], [0.9, 0.5, 0.2]], 
                [1, 0],
                [[[0.302064, 0.247309], [0.390694, 0.319873]], [[0.488903, 0.242782], [0.309344, 0.461488]]]
            ),
        ]
    )
    def test_probs_multiple_calls_accumulate_and_clears_upon_reset(
        self, input_id_1, scores_1, input_id_2, scores_2, label_index, result
    ):
        logit_processor = HFLogitProcessor(label_index=label_index) 
        for input_ids, scores in [(input_id_1, scores_1), (input_id_2, scores_2)]:
            logit_processor(
                torch.tensor(input_ids),
                torch.tensor(scores)
            )
        stored_scores = logit_processor.get_probs()
        logit_processor.reset()
        post_reset_score = logit_processor.get_probs()

        # expected values are written (steps, batch, labels); get_probs is (batch, steps, labels)
        assert torch.allclose(
            stored_scores,
            torch.tensor(result).transpose(0, 1),
            atol=1e-6
        )
        assert torch.equal(
            post_reset_score,
            torch.tensor([])
        )

    def test_empty_label_index_raises_error(self):
        with pytest.raises(ValueError):
            HFLogitProcessor(label_index=[])

    @pytest.mark.parametrize("label_index", [[3], [0, 3], [0, 100]])
    def test_out_of_range_label_index_raises_error(self, label_index):
        logit_processor = HFLogitProcessor(label_index=label_index)

        with pytest.raises(IndexError):
            logit_processor(torch.tensor([[1, 2]]), torch.tensor([[0.2, 0.4, 0.8]]))

    def test_get_probs_before_any_call_is_empty(self):
        logit_processor = HFLogitProcessor(label_index=[0, 1])

        assert torch.equal(logit_processor.get_probs(), torch.tensor([]))

    def test_stored_scores_unaffected_by_later_changes_to_input(self):
        # Arrange
        logit_processor = HFLogitProcessor(label_index=[0, 1])
        scores = torch.tensor([[0.2, 0.4, 0.8]])

        # Act
        logit_processor(torch.tensor([[1, 2]]), scores)
        scores.zero_()

        # Assert
        assert torch.allclose(logit_processor.get_probs(), torch.tensor([[[0.247309, 0.302064]]]), atol=1e-6)

    def test_probs_are_normalised_over_full_vocab_not_just_labels(self):
        # Arrange
        scores = torch.tensor([[0.2, 0.4, 0.8], [0.3, 0.5, 0.2]])
        full = HFLogitProcessor(label_index=[0, 1, 2])
        subset = HFLogitProcessor(label_index=[0, 1])

        # Act
        full(torch.tensor([[1], [2]]), scores)
        subset(torch.tensor([[1], [2]]), scores)

        # Assert
        assert torch.allclose(full.get_probs().sum(dim=-1), torch.ones(2, 1))
        assert (subset.get_probs().sum(dim=-1) < 1).all()


def _fake_dataset(n_train=100, n_test=20):
    return DatasetDict({
        "train": Dataset.from_dict({"id": list(range(n_train))}),
        "test": Dataset.from_dict({"id": list(range(1000, 1000 + n_test))}),
    })


@pytest.fixture
def fake_load_dataset():
    with patch("shared.hf_utils.datasets.load_dataset", side_effect=lambda name: _fake_dataset()) as mock:
        yield mock


@pytest.mark.usefixtures("fake_load_dataset")
class TestLoadHFDataset:

    @pytest.mark.parametrize("train_split", [0.0, -0.5, 1.5, float("nan")])
    def test_out_of_range_train_split_raises_value_error(self, train_split):
        with pytest.raises(ValueError):
            load_hf_dataset("dummy", train_split, seed=0)

    @pytest.mark.parametrize("train_split", [1, "0.8", None, [0.8]])
    def test_non_float_train_split_raises_type_error(self, train_split):
        with pytest.raises(TypeError):
            load_hf_dataset("dummy", train_split, seed=0)

    @pytest.mark.parametrize(
        "train_split, n_train, n_val",
        [(0.8, 80, 20), (0.5, 50, 50), (0.25, 25, 75), (1.0, 100, None)],  
    )
    def test_split_sizes(self, train_split, n_train, n_val):
        splits = load_hf_dataset("dummy", train_split, seed=0)

        assert len(splits["train"]) == n_train
        assert len(splits["test"]) == 20
        assert len(splits["val"]) == n_val if n_val else splits["val"] is None 

    def test_split_is_identical_across_repeated_calls(self):
        first = load_hf_dataset("dummy", 0.8, seed=42)

        for _ in range(3):
            other = load_hf_dataset("dummy", 0.8, seed=42)
            assert first["train"]["id"] == other["train"]["id"]
            assert first["val"]["id"] == other["val"]["id"]
            assert first["test"]["id"] == other["test"]["id"]

    def test_different_seeds_give_different_splits(self):
        a = load_hf_dataset("dummy", 0.8, seed=1)
        b = load_hf_dataset("dummy", 0.8, seed=2)

        assert a["train"]["id"] != b["train"]["id"]
