"""Evaluate val_best.pt on held-out 20%, or all test_samples with --final, without training."""

import argparse
from pathlib import Path

from filelock import Timeout

from mcpp.vae_evaluation import evaluate_test_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="val_best.pt or fold_N.val_best.pt in its original experiment directory")
    parser.add_argument("--final", action="store_true", help="Evaluate all external test_samples patients")
    parser.add_argument("--data-dir", type=Path,
                        help="Default: data/training_data, or data/test_samples with --final")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <experiment>/test_evaluation/<checkpoint name>/; final_evaluation/ with --final")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.data_dir is None:
        args.data_dir = ROOT / "data" / ("test_samples" if args.final else "training_data")
    try:
        report, output = evaluate_test_checkpoint(
            args.checkpoint, args.data_dir, output_dir=args.output_dir,
            device=args.device, batch_size=args.batch_size, cpu_threads=args.cpu_threads, final=args.final)
    except Timeout:
        parser.error("Another evaluation is writing to this output directory.")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    m = report["metrics"]
    headline = "Full test_samples" if args.final else "Held-out test"
    print(f"\n{headline} MI classification: {report['n_patients']} scored / "
          f"{report['n_test_patients']} test patients; {report['n_quality_excluded']} quality exclusions\n"
          f"Checkpoint: {report['checkpoint']} (epoch {report['checkpoint_epoch']})\n"
          f"Device: {report['device']}\n"
          f"F1 (pMI): {m['f1_pMI']:.4f}\n"
          f"Accuracy: {m['accuracy']:.4f} ({100 * m['accuracy']:.2f}%)\n"
          f"Macro F1: {m['macro_f1']:.4f}\n"
          f"Balanced accuracy: {m['balanced_accuracy']:.4f}\n"
          f"ROC-AUC: {m['roc_auc']:.4f}\n"
          f"Report: {output / 'mi_classification_report.json'}\n"
          f"Patient predictions: {output / 'mi_predictions.csv'}\n"
          f"Frame-quality audit: {output / 'frame_quality.json'}", flush=True)


if __name__ == "__main__":
    main()
