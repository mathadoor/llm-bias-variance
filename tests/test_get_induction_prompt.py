import pytest
from datasets import Dataset

from scripts.get_induction_prompt import create_runs


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
