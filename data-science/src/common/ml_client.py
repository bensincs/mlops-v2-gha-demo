"""Utility for constructing an Azure ML MLClient using job environment variables.

Relies on the standard AzureML injected environment variables inside a job:
  AZUREML_ARM_SUBSCRIPTION
  AZUREML_ARM_RESOURCEGROUP
  AZUREML_ARM_WORKSPACE_NAME
"""

import os
import sys
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.ai.ml import MLClient


def get_ml_client():
    subscription_id = os.getenv("AZUREML_ARM_SUBSCRIPTION")
    resource_group = os.getenv("AZUREML_ARM_RESOURCEGROUP")
    workspace_name = os.getenv("AZUREML_ARM_WORKSPACE_NAME")
    if not all([subscription_id, resource_group, workspace_name]):
        raise EnvironmentError(
            "Missing one or more AzureML workspace environment variables: "
            "AZUREML_ARM_SUBSCRIPTION, AZUREML_ARM_RESOURCEGROUP, AZUREML_ARM_WORKSPACE_NAME"
        )
    try:
        cred = DefaultAzureCredential(exclude_interactive_browser_credential=True)
        # Probe token quickly (management scope)
        cred.get_token("https://management.azure.com/.default")
        print("Acquired token using DefaultAzureCredential", file=sys.stderr)
    except Exception as e:
        print(f"DefaultAzureCredential failed: {e}; attempting ManagedIdentityCredential", file=sys.stderr)
        cred = ManagedIdentityCredential()
    return MLClient(cred, subscription_id, resource_group, workspace_name)
