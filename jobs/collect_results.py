#!/usr/bin/env python3
"""
Collect and display model evaluation results from JSON files.
"""

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def collect_model_results(models_dir: Path) -> pd.DataFrame:
    """
    Collect evaluation results from all JSON files in the models directory.

    Args:
        models_dir: Path to the directory containing model JSON files

    Returns:
        DataFrame with model results
    """
    results = []

    # Find all JSON files in the specified directory only
    json_files = sorted(models_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {models_dir}")
        return pd.DataFrame()

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Extract hyperparameters as a formatted string
            hyperparams = data.get("hyperparameters", {})
            hyperparam_str = ", ".join([f"{k}={v}" for k, v in hyperparams.items()])

            # Extract evaluation metrics
            evaluation = data.get("evaluation", {})
            cm = evaluation.get("confusion_matrix")

            result = {
                "Model File": json_file.stem,
                "Model Type": data.get("model", "Unknown"),
                "Hyperparameters": hyperparam_str,
                "Balanced Accuracy": evaluation.get("balanced_accuracy", None),
                "Accuracy": evaluation.get("accuracy", None),
                "Recall": evaluation.get("recall", None),
                "F1 Macro": evaluation.get("f1_macro", None),
                "TN": cm[0][0] if cm else None,
                "FP": cm[0][1] if cm else None,
                "FN": cm[1][0] if cm else None,
                "TP": cm[1][1] if cm else None,
                "Saved At": data.get("saved_at", "Unknown"),
                "Num Records": len(data.get("records", [])),
            }

            results.append(result)

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(results)

    return df


def display_results(
    df: pd.DataFrame, sort_by: str = "F1 Macro", ascending: bool = False
):
    """
    Display the results in a formatted table.

    Args:
        df: DataFrame with model results
        sort_by: Column to sort by
        ascending: Sort order
    """
    if df.empty:
        print("No results to display")
        return

    # Sort the DataFrame
    if sort_by in df.columns and df[sort_by].notna().any():
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Format numeric columns to 4 decimal places
    for col in ["Balanced Accuracy", "Accuracy", "Recall", "F1 Macro"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    # Display the table
    print("\n" + "=" * 120)
    print("MODEL EVALUATION RESULTS")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120)

    # Display summary statistics
    print("\nSUMMARY:")
    print(f"  Total models evaluated: {len(df)}")

    # Group by model type
    if "Model Type" in df.columns:
        print("\n  By Model Type:")
        type_counts = df["Model Type"].value_counts()
        for model_type, count in type_counts.items():
            print(f"    - {model_type}: {count} models")


def save_to_csv(df: pd.DataFrame, output_file: Path):
    """
    Save results to a CSV file.

    Args:
        df: DataFrame with model results
        output_file: Path to output CSV file
    """
    # Create parent directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


def main():
    """Main function to collect and display model results."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect and display model evaluation results"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "models",
        help="Directory containing model JSON files (default: data/models)",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="F1 Macro",
        choices=[
            "Balanced Accuracy",
            "Accuracy",
            "Recall",
            "F1 Macro",
            "Model Type",
            "Model File",
        ],
        help="Column to sort results by (default: F1 Macro)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: descending)",
    )
    parser.add_argument("--output", type=Path, help="Save results to CSV file")
    parser.add_argument(
        "--filter-model",
        type=str,
        help="Filter by model type (e.g., KNN, SVM, RandomForest)",
    )

    args = parser.parse_args()

    # Collect results
    print(f"Collecting results from: {args.models_dir}")
    df = collect_model_results(args.models_dir)

    if df.empty:
        return

    # Apply filter if specified
    if args.filter_model:
        df = df[df["Model Type"].str.contains(args.filter_model, case=False, na=False)]
        print(f"Filtered to {args.filter_model} models: {len(df)} results")

    # Display results
    display_results(df, sort_by=args.sort_by, ascending=args.ascending)

    # Save to CSV if requested
    if args.output:
        save_to_csv(df, args.output)


if __name__ == "__main__":
    main()
