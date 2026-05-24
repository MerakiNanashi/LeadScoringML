"""Example dataset preparation workflow.

Run from the repository root:

    python -m src.data.example_usage
"""

from src.data import (
    build_feature_matrix,
    make_semisynthetic_leads,
    train_test_split_prepared,
)


def main() -> None:
    data = make_semisynthetic_leads(n=10_000, seed=7)
    train, test = train_test_split_prepared(data, test_size=0.25)
    X_train, X_test, preprocessor = build_feature_matrix(train, test)

    print("Rows:", len(data.frame))
    print("Train/Test:", X_train.shape, X_test.shape)
    print("Outcome rate:", round(data.frame["outcome"].mean(), 4))
    print("Treatment rate:", round(data.frame["treatment"].mean(), 4))
    print("Average oracle uplift:", round(data.frame["tau"].mean(), 4))
    print("Encoded features:", X_train.shape[1])


if __name__ == "__main__":
    main()
