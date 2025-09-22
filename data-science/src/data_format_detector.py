#!/usr/bin/env python3
"""
Data Format Detector Component
Automatically detect the format of input dataset (COCO, YOLO, etc.)
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_data_format(input_data: str, format_info: str, format_report: str):
    """
    Detect the format of input dataset.

    Args:
        input_data: Input dataset folder path
        format_info: Output file path for format detection results
        format_report: Output file path for detailed format analysis report
    """

    logger.info("Starting data format detection...")
    logger.info(f"Input data path: {input_data}")
    logger.info(f"Format info output: {format_info}")
    logger.info(f"Format report output: {format_report}")

    input_path = Path(input_data)

    # Validate input path
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        # Create default fallback results
        detection_results = {
            "detected_format": "unknown",
            "confidence": 0.0,
            "format_indicators": {"error": "Input path not found"},
            "file_analysis": {},
            "conversion_needed": True,
            "skip_conversion": False,
        }
        detailed_report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "input_path": str(input_path),
            "files_analyzed": [],
            "detection_logic": ["Input path not found"],
            "error": "Input path does not exist",
        }

        # Write fallback results
        try:
            os.makedirs(os.path.dirname(format_info), exist_ok=True)
            with open(format_info, "w") as f:
                json.dump(detection_results, f, indent=2)
            os.makedirs(os.path.dirname(format_report), exist_ok=True)
            with open(format_report, "w") as f:
                json.dump(detailed_report, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write fallback results: {e}")
        return

    logger.info(f"Analyzing data format in: {input_path}")

    # Initialize detection results
    detection_results = {
        "detected_format": "unknown",
        "confidence": 0.0,
        "format_indicators": {},
        "file_analysis": {},
        "conversion_needed": True,
        "skip_conversion": False,
    }

    detailed_report = {
        "analysis_timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "files_analyzed": [],
        "detection_logic": [],
    }

    try:
        # Check for COCO format indicators
        coco_indicators = 0
        yolo_indicators = 0

        logger.info("Checking for file structure indicators...")

        # Get list of files with error handling
        try:
            all_files = list(input_path.rglob("*"))
            detailed_report["files_analyzed"] = [
                str(f) for f in all_files[:20]
            ]  # Limit for report size
            logger.info(f"Found {len(all_files)} files to analyze")
        except Exception as e:
            logger.warning(f"Error listing files: {e}")
            all_files = []

        # Look for COCO annotation files (JSON)
        json_files = list(input_path.glob("*.json")) + list(
            input_path.glob("**/*.json")
        )
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Check for COCO structure
                if isinstance(data, dict) and all(
                    key in data for key in ["images", "annotations", "categories"]
                ):
                    coco_indicators += 3
                    detailed_report["detection_logic"].append(
                        f"Found COCO structure in {json_file.name}"
                    )
                    detailed_report["files_analyzed"].append(str(json_file))
                elif isinstance(data, dict) and any(
                    key in data for key in ["images", "annotations"]
                ):
                    coco_indicators += 1
                    detailed_report["detection_logic"].append(
                        f"Found partial COCO structure in {json_file.name}"
                    )

            except Exception as e:
                logger.warning(f"Could not parse JSON file {json_file}: {e}")

        # Look for YOLO format indicators
        txt_files = list(input_path.glob("*.txt")) + list(input_path.glob("**/*.txt"))
        classes_file = None

        # Check for classes.txt or similar
        for txt_file in txt_files:
            if txt_file.name.lower() in ["classes.txt", "class.names", "obj.names"]:
                classes_file = txt_file
                yolo_indicators += 2
                detailed_report["detection_logic"].append(
                    f"Found YOLO classes file: {txt_file.name}"
                )
                detailed_report["files_analyzed"].append(str(txt_file))

        # Check for YOLO annotation files (txt files with normalized coordinates)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"**/*{ext}"))

        yolo_annotation_count = 0
        for img_file in image_files[:10]:  # Check first 10 images
            annotation_file = img_file.with_suffix(".txt")
            if annotation_file.exists():
                try:
                    with open(annotation_file, "r") as f:
                        lines = f.readlines()

                    # Check YOLO format (class_id x_center y_center width height)
                    valid_yolo_lines = 0
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                # Check if coordinates are normalized (0-1 range)
                                coords = [float(x) for x in parts[1:5]]
                                if all(0 <= coord <= 1 for coord in coords):
                                    valid_yolo_lines += 1
                            except ValueError:
                                continue

                    if valid_yolo_lines > 0:
                        yolo_annotation_count += 1

                except Exception as e:
                    logger.warning(
                        f"Could not read annotation file {annotation_file}: {e}"
                    )

        if yolo_annotation_count >= 3:  # If at least 3 images have YOLO annotations
            yolo_indicators += 3
            detailed_report["detection_logic"].append(
                f"Found {yolo_annotation_count} files with YOLO format annotations"
            )
        elif yolo_annotation_count > 0:
            yolo_indicators += 1
            detailed_report["detection_logic"].append(
                f"Found {yolo_annotation_count} files with possible YOLO annotations"
            )

        # Determine format based on indicators
        if coco_indicators >= yolo_indicators and coco_indicators >= 2:
            detection_results["detected_format"] = "coco"
            detection_results["confidence"] = min(0.9, 0.3 + (coco_indicators * 0.15))
            detection_results["conversion_needed"] = True
            detection_results["skip_conversion"] = False
            logger.info(
                f"Detected COCO format with confidence {detection_results['confidence']:.2f}"
            )
        elif yolo_indicators >= 2:
            detection_results["detected_format"] = "yolo"
            detection_results["confidence"] = min(0.9, 0.3 + (yolo_indicators * 0.15))
            detection_results["conversion_needed"] = False
            detection_results["skip_conversion"] = True
            logger.info(
                f"Detected YOLO format with confidence {detection_results['confidence']:.2f}"
            )
        else:
            detection_results["detected_format"] = "unknown"
            detection_results["confidence"] = 0.0
            detection_results["conversion_needed"] = True
            detection_results["skip_conversion"] = False
            logger.warning("Could not reliably detect data format")

        # Store analysis details
        detection_results["format_indicators"] = {
            "coco_score": coco_indicators,
            "yolo_score": yolo_indicators,
            "json_files_found": len(json_files),
            "txt_files_found": len(txt_files),
            "image_files_found": len(image_files),
            "classes_file_found": classes_file is not None,
        }

        detection_results["file_analysis"] = {
            "total_json_files": len(json_files),
            "total_txt_files": len(txt_files),
            "total_image_files": len(image_files),
            "yolo_annotation_files": yolo_annotation_count,
            "classes_file_path": str(classes_file) if classes_file else None,
        }

    except Exception as e:
        logger.error(f"Error during format detection: {e}")
        detection_results["detected_format"] = "error"
        detection_results["error"] = str(e)

    # Write format detection results with error handling
    try:
        os.makedirs(os.path.dirname(format_info), exist_ok=True)
        with open(format_info, "w") as f:
            json.dump(detection_results, f, indent=2)
        logger.info(f"Successfully wrote format info to: {format_info}")
    except Exception as e:
        logger.error(f"Failed to write format info: {e}")

    # Write detailed report with error handling
    try:
        os.makedirs(os.path.dirname(format_report), exist_ok=True)
        with open(format_report, "w") as f:
            json.dump(detailed_report, f, indent=2)
        logger.info(f"Successfully wrote format report to: {format_report}")
    except Exception as e:
        logger.error(f"Failed to write format report: {e}")

    logger.info(
        f"Format detection complete: {detection_results['detected_format']} "
        f"(confidence: {detection_results['confidence']:.2f})"
    )


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Detect data format (COCO/YOLO)")
    parser.add_argument("--input-data", required=True, help="Input dataset folder")
    parser.add_argument("--format-info", required=True, help="Format info output file")
    parser.add_argument(
        "--format-report", required=True, help="Detailed report output file"
    )

    args = parser.parse_args()
    detect_data_format(args.input_data, args.format_info, args.format_report)


if __name__ == "__main__":
    main()
