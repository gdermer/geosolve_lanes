import argparse
import sys

import pandas as pd
import fiftyone as fo

DATASET_NAME = "lane_review"


def load_dataset(csv_path: str) -> fo.Dataset:
    """Load the CSV into a FiftyOne dataset. Deletes any existing dataset
    with the same name first, so re-running 'review' starts fresh from
    the CSV rather than layering on top of a stale in-memory dataset."""

    df = pd.read_csv(csv_path)

    required_cols = {"Filepath", "Lane_AI", "AI_Confidence", "Needs_review"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"WARNING: CSV is missing expected columns: {missing}")

    if fo.dataset_exists(DATASET_NAME):
        fo.delete_dataset(DATASET_NAME)

    dataset = fo.Dataset(DATASET_NAME)

    samples = []
    for _, row in df.iterrows():
        sample = fo.Sample(filepath=row["Filepath"])

        if "Lane_AI" in df.columns and pd.notna(row.get("Lane_AI")):
            confidence = row.get("AI_Confidence")
            sample["predicted_label"] = fo.Classification(
                label=str(row["Lane_AI"]),
                confidence=float(confidence) if pd.notna(confidence) else None,
            )

        if "Needs_review" in df.columns:
            nr = row.get("Needs_review")
            sample["needs_review"] = bool(nr) if pd.notna(nr) else False

        if "_session" in df.columns:
            sample["source_session"] = row.get("_session")

        if "Bearing" in df.columns:
            sample["bearing"] = row.get("Bearing")

        samples.append(sample)

    dataset.add_samples(samples)
    dataset.persistent = True  # survives between script runs
    print(f"Loaded {len(dataset)} samples into dataset '{DATASET_NAME}'")
    return dataset


def run_review(csv_path: str):
    """Load the CSV and launch the FiftyOne App for browsing/correcting."""
    dataset = load_dataset(csv_path)
    session = fo.launch_app(dataset)
    print("FiftyOne App launched. Correct labels in the UI by setting the")
    print("'human_label' field on each sample (via the app's sample panel).")
    print("Press Ctrl+C in this terminal when you're done reviewing.")
    session.wait()  # blocks until you close the app / hit Ctrl+C


def export_corrections(csv_path: str, output_path: str):
    """Merge human_label corrections (from the FiftyOne dataset) back into
    the original CSV and write the result to output_path."""

    if not fo.dataset_exists(DATASET_NAME):
        print(f"ERROR: dataset '{DATASET_NAME}' not found. Run 'review' first.")
        sys.exit(1)

    dataset = fo.load_dataset(DATASET_NAME)
    df = pd.read_csv(csv_path)

    records = []
    for sample in dataset:
        human_label = None
        if sample.has_field("human_label") and sample["human_label"] is not None:
            human_label = sample["human_label"].label

        records.append({
            "Filepath": sample.filepath,
            "human_label": human_label,
        })

    corrections_df = pd.DataFrame(records)
    merged = df.merge(corrections_df, on="Filepath", how="left")
    merged.to_csv(output_path, index=False)
    print(f"Wrote {len(merged)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Review and correct lane labels with FiftyOne.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    review_parser = subparsers.add_parser("review", help="Load CSV and launch the FiftyOne App")
    review_parser.add_argument("csv_path", help="Path to the input CSV file")

    export_parser = subparsers.add_parser("export", help="Export corrected labels to a new CSV")
    export_parser.add_argument("csv_path", help="Path to the original input CSV file")
    export_parser.add_argument("output_path", help="Path to write the corrected CSV to")

    args = parser.parse_args()

    if args.command == "review":
        run_review(args.csv_path)
    elif args.command == "export":
        export_corrections(args.csv_path, args.output_path)


if __name__ == "__main__":
    main()