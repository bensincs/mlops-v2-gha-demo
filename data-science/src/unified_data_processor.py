#!/usr/bin/env python3
"""
Unified Data Processor Component
Handles both COCO to YOLO conversion and direct YOLO copying in a single component
"""

import os
import json
import logging
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_data(
    input_data: str, detected_format: str, output_data: str, processing_report: str
):
    """
    Process data based on detected format - either convert COCO to YOLO or copy YOLO directly.

    Args:
        input_data: Input dataset folder path
        detected_format: Format detection results file path
        output_data: Output folder path
        processing_report: Processing report output file
    """

    logger.info("Starting unified data processing...")
    logger.info(f"Input data: {input_data}")
    logger.info(f"Detected format file: {detected_format}")
    logger.info(f"Output data: {output_data}")
    logger.info(f"Processing report: {processing_report}")

    # Load format detection results with error handling
    format_info = {}
    try:
        if not os.path.exists(detected_format):
            logger.warning(f"Format detection file not found: {detected_format}")
            format_info = {
                "detected_format": "unknown",
                "confidence": 0.0,
                "conversion_needed": True,
            }
        else:
            with open(detected_format, "r") as f:
                format_info = json.load(f)
            logger.info(f"Loaded format info: {format_info}")
    except Exception as e:
        logger.error(f"Error reading format detection file: {e}")
        format_info = {
            "detected_format": "unknown",
            "confidence": 0.0,
            "conversion_needed": True,
        }

    input_path = Path(input_data)
    output_path = Path(output_data)

    # Initialize processing results
    processing_results = {
        "processing_type": "none",
        "success": False,
        "detected_format": format_info.get("detected_format", "unknown"),
        "files_processed": 0,
        "errors": [],
        "warnings": [],
        "timestamp": datetime.now().isoformat(),
    }

    # Validate input path
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        processing_results["errors"].append("Input path not found")
        # Create empty output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Write processing report
        with open(processing_report, "w") as f:
            json.dump(processing_results, f, indent=2)
        return

    # Determine processing type based on detected format
    detected_format_type = format_info.get("detected_format", "unknown")

    if detected_format_type == "coco":
        logger.info("Detected COCO format - performing conversion to YOLO")
        processing_results["processing_type"] = "coco_to_yolo_conversion"
        success = convert_coco_to_yolo(input_path, output_path, processing_results)
        processing_results["success"] = success

    elif detected_format_type == "yolo":
        logger.info("Detected YOLO format - copying data directly")
        processing_results["processing_type"] = "yolo_direct_copy"
        success = copy_yolo_data_directly(input_path, output_path, processing_results)
        processing_results["success"] = success

    else:
        logger.warning(f"Unknown or unsupported format: {detected_format_type}")
        processing_results["processing_type"] = "unknown_format"
        processing_results["warnings"].append(
            f"Unsupported format: {detected_format_type}"
        )
        # Create empty output directory
        output_path.mkdir(parents=True, exist_ok=True)
        processing_results["success"] = True  # Not a failure, just no processing needed

    # Write processing report
    try:
        os.makedirs(os.path.dirname(processing_report), exist_ok=True)
        with open(processing_report, "w") as f:
            json.dump(processing_results, f, indent=2)
        logger.info(f"Processing report written to: {processing_report}")
    except Exception as e:
        logger.error(f"Failed to write processing report: {e}")

    logger.info("Unified data processing completed")


def convert_coco_to_yolo(input_path: Path, output_path: Path, results: dict) -> bool:
    """Convert COCO format to YOLO format."""
    try:
        # Create output subdirectories
        output_path.mkdir(parents=True, exist_ok=True)
        images_dir = output_path / "images"
        labels_dir = output_path / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        # Find COCO annotation files
        coco_files = list(input_path.glob("*.json"))
        if not coco_files:
            logger.warning("No JSON files found in input directory")
            results["warnings"].append("No JSON files found")
            return True  # Not an error, just no data to process

        logger.info(f"Found {len(coco_files)} JSON files to process")

        # Process COCO annotation files
        all_categories = {}
        converted_annotations = 0

        for json_file in coco_files:
            try:
                with open(json_file, "r") as f:
                    coco_data = json.load(f)

                if not isinstance(coco_data, dict) or "images" not in coco_data:
                    continue

                # Build category mapping
                if "categories" in coco_data:
                    for cat in coco_data["categories"]:
                        all_categories[cat["id"]] = cat["name"]

                # Create image filename to ID mapping
                image_mapping = {}
                if "images" in coco_data:
                    for img in coco_data["images"]:
                        image_mapping[img["id"]] = {
                            "filename": img["file_name"],
                            "width": img.get("width", 640),
                            "height": img.get("height", 480),
                        }

                # Process annotations
                if "annotations" in coco_data:
                    # Group annotations by image
                    annotations_by_image = {}
                    for ann in coco_data["annotations"]:
                        img_id = ann["image_id"]
                        if img_id not in annotations_by_image:
                            annotations_by_image[img_id] = []
                        annotations_by_image[img_id].append(ann)

                    # Convert annotations to YOLO format
                    for img_id, annotations in annotations_by_image.items():
                        if img_id not in image_mapping:
                            continue

                        img_info = image_mapping[img_id]
                        img_width = img_info["width"]
                        img_height = img_info["height"]
                        filename = img_info["filename"]

                        # Create YOLO annotation file
                        yolo_filename = Path(filename).stem + ".txt"
                        yolo_path = labels_dir / yolo_filename

                        yolo_lines = []
                        for ann in annotations:
                            if "bbox" not in ann or "category_id" not in ann:
                                continue

                            # COCO bbox format: [x, y, width, height]
                            bbox = ann["bbox"]
                            x, y, w, h = bbox

                            # Convert to YOLO format (normalized center coordinates)
                            x_center = (x + w / 2) / img_width
                            y_center = (y + h / 2) / img_height
                            norm_width = w / img_width
                            norm_height = h / img_height

                            # YOLO class ID (0-based)
                            class_id = (
                                ann["category_id"] - 1 if ann["category_id"] > 0 else 0
                            )

                            yolo_lines.append(
                                f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
                            )
                            converted_annotations += 1

                        # Write YOLO annotation file
                        with open(yolo_path, "w") as f:
                            f.write("\n".join(yolo_lines))

                        results["files_processed"] += 1

            except Exception as e:
                logger.warning(f"Error processing COCO file {json_file}: {e}")
                results["errors"].append(
                    f"COCO processing error in {json_file.name}: {str(e)}"
                )

        # Copy image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"**/*{ext}"))

        copied_images = 0
        for img_file in image_files:
            try:
                shutil.copy2(img_file, images_dir / img_file.name)
                copied_images += 1
            except Exception as e:
                logger.warning(f"Error copying image {img_file}: {e}")

        # Create classes.txt file
        classes_file = output_path / "classes.txt"
        with open(classes_file, "w") as f:
            if all_categories:
                # Write categories in order of their IDs
                sorted_categories = sorted(all_categories.items())
                for cat_id, cat_name in sorted_categories:
                    f.write(f"{cat_name}\n")
            else:
                # Default class if no categories found
                f.write("object\n")

        results["annotations_converted"] = converted_annotations
        results["images_copied"] = copied_images

        logger.info(f"✅ COCO to YOLO conversion completed!")
        logger.info(f"   - Processed {results['files_processed']} annotation files")
        logger.info(f"   - Converted {converted_annotations} annotations")
        logger.info(f"   - Copied {copied_images} images")

        return True

    except Exception as e:
        logger.error(f"Error during COCO to YOLO conversion: {e}")
        results["errors"].append(f"Conversion error: {str(e)}")
        # Create empty output to continue pipeline
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "images").mkdir(exist_ok=True)
        (output_path / "labels").mkdir(exist_ok=True)
        return False


def copy_yolo_data_directly(input_path: Path, output_path: Path, results: dict) -> bool:
    """Copy YOLO format data directly."""
    try:
        # Copy the entire input directory to output
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(input_path, output_path)

        # Verify the copy was successful
        try:
            input_files = len(list(input_path.rglob("*")))
            output_files = len(list(output_path.rglob("*")))

            logger.info(f"✅ Successfully copied YOLO data to {output_path}")
            logger.info(f"   - Input files: {input_files}")
            logger.info(f"   - Output files: {output_files}")

            results["files_processed"] = input_files
            results["files_copied"] = output_files

            if input_files != output_files:
                logger.warning(
                    f"File count mismatch: input={input_files}, output={output_files}"
                )
                results["warnings"].append(
                    f"File count mismatch: input={input_files}, output={output_files}"
                )

        except Exception as verify_error:
            logger.warning(f"Could not verify file counts: {verify_error}")
            results["warnings"].append(
                f"Could not verify file counts: {str(verify_error)}"
            )

        return True

    except Exception as e:
        logger.error(f"Error copying YOLO data: {e}")
        results["errors"].append(f"Copy error: {str(e)}")
        # Create empty output directory to prevent pipeline failure
        output_path.mkdir(parents=True, exist_ok=True)
        return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Unified data processor for COCO/YOLO formats"
    )
    parser.add_argument("--input-data", required=True, help="Input dataset folder")
    parser.add_argument(
        "--detected-format", required=True, help="Format detection results file"
    )
    parser.add_argument("--output-data", required=True, help="Output folder")
    parser.add_argument(
        "--processing-report", required=True, help="Processing report output file"
    )

    args = parser.parse_args()
    process_data(
        args.input_data, args.detected_format, args.output_data, args.processing_report
    )


if __name__ == "__main__":
    main()
