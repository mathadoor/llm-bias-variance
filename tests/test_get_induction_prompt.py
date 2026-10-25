from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from datasets import Dataset

from scripts.get_induction_prompt import create_runs, generate_induction_prompts


@pytest.fixture
def train_dataset():
    return Dataset.from_dict({"id": list(range(50))})


def _ids(run):
    return [row["id"] for row in run.ravel()]


class TestCreateRuns:

    @pytest.mark.parametrize(
        "train_sizes, num_runs",
        [([], 3), ([5, 10, 20], 3), ([10], 1), ([1], 4)],
    )
    def test_one_entry_per_train_size_with_one_column_per_run(self, train_dataset, train_sizes, num_runs):
        runs = create_runs(train_dataset, train_sizes=train_sizes, num_runs_per_size=num_runs, seed=0)

        assert [run["train_size"] for run in runs] == train_sizes
        assert [run["dataset"].shape for run in runs] == [(size, num_runs) for size in train_sizes]

    def test_samples_come_from_the_train_dataset(self, train_dataset):
        runs = create_runs(train_dataset, train_sizes=[10, 20], num_runs_per_size=3, seed=0)

        valid_ids = set(train_dataset["id"])
        for run in runs:
            assert set(_ids(run["dataset"])) <= valid_ids

    @pytest.mark.parametrize("seed_a, seed_b, identical", [(42, 42, True), (1, 2, False)])
    def test_runs_are_reproducible_only_for_the_same_seed(self, train_dataset, seed_a, seed_b, identical):
        a = create_runs(train_dataset, train_sizes=[5, 10], num_runs_per_size=3, seed=seed_a)
        b = create_runs(train_dataset, train_sizes=[5, 10], num_runs_per_size=3, seed=seed_b)

        ids_a = [_ids(run["dataset"]) for run in a]
        ids_b = [_ids(run["dataset"]) for run in b]
        assert (ids_a == ids_b) is identical

    def test_runs_within_a_size_are_different_samples(self, train_dataset):
        run = create_runs(train_dataset, train_sizes=[10], num_runs_per_size=3, seed=0)[0]["dataset"]

        # columns are the runs
        runs = [[row["id"] for row in samples] for samples in run.T]
        assert len({tuple(r) for r in runs}) == len(runs)



def _make_runs(train_sizes, num_runs):
    # same layout as create_runs: (train_size, num_runs), each column is one run
    runs = []
    for size in train_sizes:
        data = np.empty((size, num_runs), dtype=object)
        for row in range(size):
            for col in range(num_runs):
                data[row, col] = {"id": 1000 * size + 100 * col + row}
        runs.append({"train_size": size, "dataset": data})
    return runs


def _run_ids(run):
    return [tuple(sample["id"] for sample in column) for column in run["dataset"].T]


@pytest.fixture
def fake_module():
    calls = []

    def module(samples):
        calls.append(samples)
        return SimpleNamespace(output_prompt="prompt-" + "-".join(str(s["id"]) for s in samples))

    with patch("scripts.get_induction_prompt.dspy.ChainOfThought", return_value=module):
        yield calls


@pytest.fixture
def fake_context():
    used = {}

    @contextmanager
    def context(**kwargs):
        used.update(kwargs)
        yield

    with patch("scripts.get_induction_prompt.dspy.context", side_effect=context):
        yield used


@pytest.mark.usefixtures("fake_context")
class TestGenerateInductionPrompts:

    def test_no_runs_returns_no_results(self, fake_module):
        results = generate_induction_prompts([], lm=object())

        assert results == []
        assert fake_module == []

    def test_one_result_per_run(self, fake_module):
        runs = _make_runs(train_sizes=[2, 3], num_runs=4)

        results = generate_induction_prompts(runs, lm=object())

        assert len(results) == 8
        assert sorted(r["train_size"] for r in results) == [2] * 4 + [3] * 4

    def test_module_receives_the_samples_of_each_run(self, fake_module):
        runs = _make_runs(train_sizes=[2, 3], num_runs=4)

        generate_induction_prompts(runs, lm=object())

        expected = {ids for run in runs for ids in _run_ids(run)}
        assert {tuple(s["id"] for s in call) for call in fake_module} == expected
        assert all(isinstance(call, list) for call in fake_module)

    def test_each_result_pairs_train_size_samples_and_module_output(self, fake_module):
        runs = _make_runs(train_sizes=[2, 3], num_runs=2)

        results = generate_induction_prompts(runs, lm=object())

        for result in results:
            ids = tuple(s["id"] for s in result["dataset"])
            assert len(ids) == result["train_size"]
            assert result["results"].output_prompt == "prompt-" + "-".join(map(str, ids))

    def test_uses_the_provided_lm(self, fake_module, fake_context):
        lm = object()

        generate_induction_prompts(_make_runs([2], 1), lm=lm)

        assert fake_context["lm"] is lm

    def test_module_error_propagates(self):
        def failing(samples):
            raise RuntimeError("lm failed")

        with patch("scripts.get_induction_prompt.dspy.ChainOfThought", return_value=failing):
            with pytest.raises(RuntimeError, match="lm failed"):
                generate_induction_prompts(_make_runs([2], 1), lm=object())
