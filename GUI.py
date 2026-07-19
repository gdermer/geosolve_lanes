"""
lane_review.py

Load a CSV of image paths + AI-predicted lane labels into FiftyOne,
browse/correct them in the FiftyOne App, then export corrections back
to a new CSV.

Usage:
    # Load the CSV and launch the browser app (blocks until you close it)
    python lane_review.py review path/to/your_file.csv

    # After you've made corrections in the app, export them
    python lane_review.py export path/to/your_file.csv path/to/corrected_output.csv

Notes:
    - The "review" and "export" steps use the same FiftyOne dataset name
      ("lane_review"), so you can run "review" to open the app, correct
      labels in the UI, leave the app running, and in a second terminal
      run "export" against the same dataset -- OR just keep everything
      in one Python session (see run_interactive() below) if you prefer
      working from a Jupyter notebook / REPL instead of the CLI.
"""

import argparse
import sys

import pandas as pd
import fiftyone as fo
from fiftyone import ViewField as F


DATASET_NAME = "lane_review"


def parse_bool(value):
    """Robustly parse a CSV cell into a real Python bool.

    Handles actual booleans, numbers (0/1), and text values like
    'True'/'False'/'TRUE'/'FALSE'/'yes'/'no' -- unlike bare bool(value),
    which treats any non-empty string (even the text 'False') as True.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in ("true", "1", "yes", "y")


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
    total = len(df)
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  building sample {idx}/{total}...")
        sample = fo.Sample(filepath=row["Filepath"])

        if "Lane_AI" in df.columns and pd.notna(row.get("Lane_AI")):
            confidence = row.get("AI_Confidence")
            sample["predicted_label"] = fo.Classification(
                label=str(row["Lane_AI"]),
                confidence=float(confidence) if pd.notna(confidence) else None,
            )

        if "Needs_review" in df.columns:
            nr = row.get("Needs_review")
            sample["needs_review"] = parse_bool(nr) if pd.notna(nr) else False

        if "_session" in df.columns:
            sample["source_session"] = row.get("_session")

        if "Bearing" in df.columns:
            sample["bearing"] = row.get("Bearing")

        samples.append(sample)

    print(f"Adding {len(samples)} samples to the dataset (this step generates thumbnails/metadata and can take a while for large datasets or slow drives)...")
    dataset.add_samples(samples)
    dataset.persistent = True  # survives between script runs
    print(f"Loaded {len(dataset)} samples into dataset '{DATASET_NAME}'")
    return dataset


def run_review(csv_path: str, share: bool = False, max_confidence: float = None,
               needs_review: bool = None):
    """Load the CSV and launch the FiftyOne App for browsing/correcting.

    If share=True, binds to 0.0.0.0 so others on the same network can
    connect via http://<your-machine-ip>:5151

    If max_confidence is set, only samples whose predicted_label.confidence
    is <= max_confidence are shown (lower confidence = more likely wrong).

    If needs_review is True or False, only samples where needs_review
    matches that value are shown. Leave as None to not filter on this field.

    Filters are combined with AND if more than one is given.
    """
    dataset = load_dataset(csv_path)

    view = dataset
    if max_confidence is not None:
        view = view.match(F("predicted_label.confidence") <= max_confidence)
        print(f"Filter applied: confidence <= {max_confidence} "
              f"({len(view)} of {len(dataset)} samples match)")
    if needs_review is not None:
        view = view.match(F("needs_review") == needs_review)
        print(f"Filter applied: needs_review == {needs_review} "
              f"({len(view)} of {len(dataset)} samples match)")

    if share:
        session = fo.launch_app(view, address="0.0.0.0", port=5151)
        print("App is shared on the network. Teammates can connect at:")
        print("  http://<YOUR_MACHINE_IP>:5151")
        print("(Find your IP with 'ipconfig' on Windows, look for IPv4 Address)")
    else:
        session = fo.launch_app(view)
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
    review_parser.add_argument("--share", action="store_true",
                                help="Bind to 0.0.0.0 so teammates on the same network can connect")
    review_parser.add_argument("--max-confidence", type=float, default=None,
                                help="Only show samples with predicted_label confidence <= this value")
    review_parser.add_argument("--needs-review", type=str, default=None, choices=["true", "false"],
                                help="Only show samples where needs_review matches true or false")

    export_parser = subparsers.add_parser("export", help="Export corrected labels to a new CSV")
    export_parser.add_argument("csv_path", help="Path to the original input CSV file")
    export_parser.add_argument("output_path", help="Path to write the corrected CSV to")

    args = parser.parse_args()

    if args.command == "review":
        needs_review = None
        if args.needs_review is not None:
            needs_review = (args.needs_review == "true")
        run_review(args.csv_path, share=args.share,
                   max_confidence=args.max_confidence,
                   needs_review=needs_review)
    elif args.command == "export":
        export_corrections(args.csv_path, args.output_path)


if __name__ == "__main__":
    main()