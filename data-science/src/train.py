#!/usr/bin/env python
"""Simple YOLOv8 training entrypoint for Azure ML pipeline.

Assumes the input dataset folder follows YOLO structure with data.yaml inside
(or user passes a path to one). Saves model artifacts to the provided output dir.
"""

import argparse
import os
import json
import logging
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--model", default="yolov8n.pt", help="Base model weights")
    p.add_argument("--output", required=True, help="Output directory for model")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Basic logging setup (stdout). Azure ML captures stdout so this will appear in job logs.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # ensure our format applies even if something configured logging earlier
    )
    logger = logging.getLogger("train")
    logger.info("===== YOLO training script start =====")
    logger.info("Output directory: %s", args.output)

    # Lazy import so env resolves dependencies at runtime
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.exception(
            "Failed to import Ultralytics YOLO (is the dependency installed?)"
        )
        raise RuntimeError("Ultralytics YOLO not installed in environment") from e

    logger.info(
        "Starting YOLO training | model=%s | data=%s | epochs=%s | imgsz=%s",
        args.model,
        args.data,
        args.epochs,
        args.imgsz,
    )

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        project=args.output,
        name="run",
    )

    # Save a lightweight metrics json for downstream steps
    metrics_path = Path(args.output) / "metrics.json"
    try:
        metrics_summary = {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "best_map50": getattr(results, "best_map50", None),
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_summary, f, indent=2)
        logger.info("Metrics saved: %s", metrics_summary)
        logger.info("Saved metrics to %s", metrics_path)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to write metrics.json: %s", e)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
