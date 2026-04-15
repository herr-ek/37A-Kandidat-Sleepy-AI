import os
from random import shuffle

DATA_LOCATION = "data/processed"
SET_LOCATION = "set_distribution"
# Distributions
TRAIN = 0.8
TEST = 0.1
VALIDATE = 0.1


def generate_sets(
    save=True,
    train: float = TRAIN,
    test: float = TEST,
    validate: float = VALIDATE,
) -> tuple[list[str], list[str], list[str]]:
    all_dirs = os.listdir(DATA_LOCATION)
    shuffle(all_dirs)

    return split_and_save_sets(
        save, all_dirs, train=train, test=test, validate=validate
    )


def split_and_save_sets(
    save,
    all_dirs: list[str],
    train: float = TRAIN,
    test: float = TEST,
    validate: float = VALIDATE,
):
    n = len(all_dirs)
    training_set = all_dirs[: int(n * train)]
    test_set = all_dirs[int(n * train) : int(n * (train + test))]
    validate_set = all_dirs[int(n * (train + test)) :]

    if save:
        with open(f"{SET_LOCATION}/training_set.txt", "w") as f:
            for patient in training_set:
                print(patient, file=f)

        with open(f"{SET_LOCATION}/test_set.txt", "w") as f:
            for patient in test_set:
                print(patient, file=f)

        with open(f"{SET_LOCATION}/validate_set.txt", "w") as f:
            for patient in validate_set:
                print(patient, file=f)

    return training_set, test_set, validate_set


def get_training_test_validate_sets() -> tuple[list[str], list[str], list[str]]:
    files = ["training_set.txt", "test_set.txt", "validate_set.txt"]

    res = []
    for file in files:
        with open(f"{SET_LOCATION}/{file}", "r") as f:
            res.append([v.strip() for v in f.readlines()])

    return tuple(res)
