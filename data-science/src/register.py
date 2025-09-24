# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Minimal model registration script using Azure ML MLClient (no MLflow).

Workspace details are inferred from AzureML job environment variables.
"""

import argparse
import os
import json
from azure.ai.ml.entities import Model
from common import get_ml_client


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_path", required=True, help="Local or mounted path to model files (dir or file)")
    p.add_argument("--model_info_output_path", required=True)
    args, _ = p.parse_known_args()
    return args


def main():
    args = parse_args()
    print(f"Registering model '{args.model_name}' from path '{args.model_path}'")

    ml_client = get_ml_client()

    model_entity = Model(
        name=args.model_name,
        path=args.model_path,  # Can be a directory or file
        description="Registered via register.py (MLClient)",
        type="custom_model",  # generic; adjust if you have a specific framework
    )

    registered = ml_client.models.create_or_update(model_entity)
    print(f"Registered model: name={registered.name} version={registered.version} id={registered.id}")

    # Persist model id in expected JSON format
    os.makedirs(args.model_info_output_path, exist_ok=True)
    out_path = os.path.join(args.model_info_output_path, "model_info.json")
    with open(out_path, "w") as f:
        json.dump({"id": f"{registered.name}:{registered.version}"}, f)
    print(f"Wrote model info to {out_path}")


if __name__ == "__main__":
    main()
