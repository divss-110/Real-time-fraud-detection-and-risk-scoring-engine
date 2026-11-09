import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from scipy.stats import ks_2samp

from src.config import (
    CURRENT_BATCH_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    REFERENCE_PATH,
    REPORTS_DIR,
    USER_FEATURES_PATH,
)
from src.features.engineering import join_user_features


def prepare_current(batch_path: Path) -> pd.DataFrame:
    batch = pd.read_parquet(batch_path)
    if "user_avg_amount_30d" not in batch.columns:
        batch = join_user_features(batch, pd.read_parquet(USER_FEATURES_PATH))
    current = batch[MODEL_FEATURES].copy()
    current["prediction"] = lgb.Booster(model_file=str(MODEL_PATH)).predict(current[MODEL_FEATURES])
    return current


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path) -> None:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
        snapshot.save_html(str(html_path))
    except ImportError:
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(html_path))


def ks_summary(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05) -> dict:
    columns = {}
    for name in reference.columns:
        stat, p_value = ks_2samp(reference[name], current[name])
        columns[name] = {
            "ks_statistic": round(float(stat), 4),
            "p_value": float(p_value),
            "drift": bool(p_value < alpha),
            "reference_mean": round(float(reference[name].mean()), 4),
            "current_mean": round(float(current[name].mean()), 4),
        }
    drifted = [name for name, info in columns.items() if info["drift"]]
    share = len(drifted) / len(columns)
    return {
        "n_reference": len(reference),
        "n_current": len(current),
        "drifted_columns": drifted,
        "share_drifted": round(share, 4),
        "dataset_drift": share >= 0.3,
        "columns": columns,
    }


def run(reference_path: Path = REFERENCE_PATH, current_path: Path = CURRENT_BATCH_PATH, output_dir: Path = REPORTS_DIR) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(reference_path)
    current = prepare_current(current_path)
    reference = reference[current.columns]
    html_path = output_dir / "drift_report.html"
    evidently_report(reference, current, html_path)
    summary = ks_summary(reference, current)
    summary["html_report"] = str(html_path)
    (output_dir / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an Evidently data drift report for a batch of transactions")
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--current", type=Path, default=CURRENT_BATCH_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    summary = run(args.reference, args.current, args.output_dir)
    print(f"Dataset drift: {summary['dataset_drift']} ({summary['share_drifted']:.0%} of columns)")
    print(f"Drifted columns: {', '.join(summary['drifted_columns']) or 'none'}")
    print(f"HTML report: {summary['html_report']}")


if __name__ == "__main__":
    main()
