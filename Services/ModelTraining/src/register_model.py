"""
__author__ = "Mateo Villa Arias"
__contributor__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.4"
__email__ = "georges.nassopoulos@gmail.com"
__status__ = "Dev"
__desc__ = "MLflow model registration with optional interactive prompts and configurable artifact path."
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
import os   
import dagshub
import mlflow

from Services.FastAPI.src.constants import (
    MLFLOW_DEFAULT_EXPERIMENT_NAME,
    MLFLOW_ENABLE_REMOTE,
    MLFLOW_LOCAL_URI,
    MLFLOW_REMOTE_URL,
    DAGSHUB_REPO_OWNER,
    DAGSHUB_REPO_NAME,
)

def display_artifacts(client: mlflow.tracking.MlflowClient, run_id: str):
    """
        Print a simple tree of top-level artifacts for a given run


        Args:
            client (mlflow.tracking.MlflowClient) : MLflow client instance
            run_id (str) : The run ID to inspect

        Returns:
            list: A list of mlflow.entities.FileInfo objects for top-level artifacts.
    """
    
    artifacts = client.list_artifacts(run_id)
    print("\nAvailable artifacts:")
    for idx, artifact in enumerate(artifacts, 1):
        print(f"{idx}. {artifact.path} {'(dir)' if artifact.is_dir else '(file)'}")
        if artifact.is_dir:
            nested = client.list_artifacts(run_id, artifact.path)
            for child in nested:
                print(f"   - {child.path}")
    return artifacts

def select_model_path(artifacts) -> str:
    """
        Let the user pick an artifact **directory** to use as model path
            - This function is used only in interactive mode
            - If there is a single directory, it is returned automatically

        Args:
            artifacts (list): List of mlflow.entities.FileInfo objects

        Returns:
            str: The selected artifact directory path
    """
    
    dirs = [a for a in artifacts if a.is_dir]

    if not dirs:
        raise Exception("No directories found in artifacts.")

    if len(dirs) == 1:
        return dirs[0].path

    print("\nMultiple model directories found. Please select one:")
    for idx, d in enumerate(dirs, 1):
        print(f"{idx}. {d.path}")

    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))
            if 1 <= choice <= len(dirs):
                return dirs[choice - 1].path
            print(f"Please enter a number between 1 and {len(dirs)}")
        except ValueError:
            print("Please enter a valid number.")

def get_model_uri(
    tracking_uri: str,
    experiment_name: str,
    run_id: str | None = None,
    filepath: str | None = None,
    interactive: bool = False,
) -> tuple[str, str]:
    """
        Build the model URI from a run ID and artifact path
            - Sets the tracking URI
            - Resolves `run_id` if not provided (latest FINISHED run)
            - If `interactive=True` and no `filepath` is provided:
                * Lists artifacts and lets the user pick a directory
              Otherwise:
                * Uses `filepath` as the artifact path (file or directory)

        Args:
            tracking_uri (str): MLflow tracking URI
            experiment_name (str): MLflow experiment name
            run_id (str): Target run ID. If None, the latest FINISHED run is used
            filepath (str): Artifact-relative path (file or directory) inside the run. If None and not interactive, defaults to "model"
            interactive(bool): Enables interactive artifact selection

        Returns:
            tuple[str, str]: A tuple of (model_uri, resolved_run_id)

    """
    
    mlflow.set_tracking_uri(tracking_uri)
    print(f"Using tracking URI: {tracking_uri}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found. Falling back to 'incident_analysis_incident_duration_min'")
        experiment_name = "incident_analysis_incident_duration_min"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise Exception(
                f"Experiment '{experiment_name}' not found either. Please check available experiments in MLflow."
            )

    # ---- FIX 1: resolve run_id properly ----
    if not run_id:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            raise Exception(
                f"No successful runs found in experiment '{experiment_name}'."
            )
        run_id = runs.iloc[0].run_id
        print(f"Found latest run ID: {run_id}")

    # ---- FIX 2: resolve filepath properly ----
    if interactive and not filepath:
        client = mlflow.tracking.MlflowClient()
        artifacts = display_artifacts(client, run_id)
        model_path = select_model_path(artifacts)
    else:
        model_path = filepath or "model"
        if os.path.isabs(model_path):
            model_path = os.path.basename(model_path)

    # DEBUG prints
    print(f"[DEBUG] run_id reçu = {run_id}")
    print(f"[DEBUG] filepath reçu = {filepath}")
    print(f"[DEBUG] model_path retenu = {model_path}")

    model_uri = f"runs:/{run_id}/{model_path}"
    print(f"Resolved model URI: {model_uri}")
    return model_uri, run_id

def register_model(model_uri: str, model_name: str, tags: dict | None = None):
    """
        Register the model in MLflow Model Registry and set optional tags

        Args:
            model_uri (str): MLflow model URI to register (e.g., 'runs:/<run_id>/model')
            model_name (str): Target registered model name
            tags (dict) : Tags to attach to the registered model.

        Returns:
            mlflow.entities.model_registry.ModelVersion: The created model version
    """
    
    print(f"\nRegistering model from: {model_uri}")
    print(f"Model name: {model_name}")

    client = mlflow.tracking.MlflowClient()
    try:
        model_details = mlflow.register_model(model_uri, model_name)
        print(f"Model registered with version: {model_details.version}")

        if tags:
            for k, v in tags.items():
                client.set_registered_model_tag(model_name, k, v)
            print("Tags set successfully.")

        return model_details
    except Exception as e:
        print("Failed to register model")
        print(f"Error: {str(e)}")
        raise

def manage_tags(model_name: str, version: str | None = None):
    """
        Interactively add, update, delete, or list tags
        Notes : Used only when `interactive=True`

        Args:
            model_name (str): Registered model name
            version (str): Version to target. If None, operates at model level
    """
    
    client = mlflow.tracking.MlflowClient()

    while True:
        print("\nTag Management Options:")
        print("1. Add/Update tag")
        print("2. Delete tag")
        print("3. List current tags")
        print("4. Exit tag management")

        choice = input("\nEnter your choice (1-4): ")

        try:
            if choice == "1":
                key = input("Enter tag key: ")
                value = input("Enter tag value: ")
                if version:
                    client.set_model_version_tag(model_name, version, key, value)
                else:
                    client.set_registered_model_tag(model_name, key, value)
                print(f"Tag {key}={value} set successfully")

            elif choice == "2":
                key = input("Enter tag key to delete: ")
                if version:
                    client.delete_model_version_tag(model_name, version, key)
                else:
                    client.delete_registered_model_tag(model_name, key)
                print(f"Tag {key} deleted successfully")

            elif choice == "3":
                if version:
                    mv = client.get_model_version(model_name, version)
                    tags = mv.tags
                else:
                    m = client.get_registered_model(model_name)
                    tags = m.tags
                print("\nCurrent tags:")
                for k, v in tags.items():
                    print(f"{k}: {v}")

            elif choice == "4":
                break

            else:
                print("Invalid choice, please try again.")

        except Exception as e:
            print(f"Error: {str(e)}")

def manage_aliases(model_name: str, version: str):
    """
        Interactively add, delete, or list aliases for a model version
            Notes : Used only when `interactive=True`

        Args:
            model_name (str): Registered model name
            version (str): Model version to manage aliases for
    """
    
    client = mlflow.tracking.MlflowClient()

    while True:
        print("\nAlias Management Options:")
        print("1. Add alias")
        print("2. Delete alias")
        print("3. List current aliases")
        print("4. Exit alias management")

        choice = input("\nEnter your choice (1-4): ")

        try:
            if choice == "1":
                alias = input(
                    f"Enter alias to add to model: {model_name}, version: {version} "
                )
                client.set_registered_model_alias(model_name, alias, version)
                print(f"Alias '{alias}' set successfully")

            elif choice == "2":
                alias = input("Enter alias to delete: ")
                client.delete_registered_model_alias(model_name, alias)
                print(f"Alias '{alias}' deleted successfully")

            elif choice == "3":
                mv = client.get_model_version(model_name, version)
                aliases = mv.aliases
                print("\nCurrent aliases:")
                for a in aliases:
                    print(f"- {a}")

            elif choice == "4":
                break

            else:
                print("Invalid choice, please try again.")

        except Exception as e:
            print(f"Error: {str(e)}")

def run_register_model(
    experiment_name: str | None = None,
    model_name: str | None = None,
    run_id: str | None = None,
    tags: str | None = None,
    filepath: str | None = None,
    interactive: bool = False,
):
    """
        Orchestrate model URI resolution and registration

            - Initializes MLflow tracking (local or DagsHub remote)
            - Parses tag string into a dict
            - If `interactive=True`, prompt for any missing values and optionally
              allow artifact directory selection and tag/alias management
            - If `interactive=False`, no `input()` is called

        Args:
            experiment_name (str): MLflow experiment name. Defaults to `MLFLOW_DEFAULT_EXPERIMENT_NAME`
            model_name (str): Registered model name. If None (non-interactive), auto-generated
            run_id (str): Run ID to target. If None, the latest FINISHED run is used
            tags (str): Comma-separated list of tags (e.g., "key1=val1,key2=val2")
            filepath (str): Artifact path (file or directory) within the run. If None (non-interactive), defaults to "model"
            interactive (bool): Enables interactive prompts and artifact selection
    """
    
    ## Initialize tracking
    if MLFLOW_ENABLE_REMOTE:
        dagshub.init(
            repo_owner=DAGSHUB_REPO_OWNER,
            repo_name=DAGSHUB_REPO_NAME,
            mlflow=True,
        )
        tracking_uri = MLFLOW_REMOTE_URL
    else:
        tracking_uri = MLFLOW_LOCAL_URI

    try:
        ## Parse tags into dict
        tag_dict = {}
        if tags:
            for pair in tags.split(","):
                key, value = pair.split("=", 1)
                tag_dict[key.strip()] = value.strip()

        ## Prompt for missing fields only if interactive
        if interactive:
            experiment_name = (
                (experiment_name or "").strip()
                or input(
                    f"Enter experiment name [{MLFLOW_DEFAULT_EXPERIMENT_NAME}]: "
                ).strip()
                or MLFLOW_DEFAULT_EXPERIMENT_NAME
            )
            run_id = (run_id or "").strip() or input(
                "Enter run id, leave blank to use the latest FINISHED run: "
            ).strip() or None
            model_name = (model_name or "").strip() or input(
                "Select a name for the model: "
            ).strip() or None
            ## Optional: let user override filepath interactively
            if not filepath:
                override = input(
                    "Artifact path (leave empty to select directory from artifacts "
                    "or use default): "
                ).strip()
                filepath = override or None

        ## Non-interactive fallbacks
        experiment_name = experiment_name or MLFLOW_DEFAULT_EXPERIMENT_NAME
        model_name = model_name or f"auto_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        ## Resolve URI
        model_uri, run_id = get_model_uri(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_id=run_id,
            filepath=filepath,
            interactive=interactive,
        )

        ## Register
        details = register_model(model_uri, model_name, tag_dict)

        ## Optional interactive post-steps
        if interactive:
            ans = input("\nManage aliases now? (yes/no): ").strip().lower()
            if ans.startswith("y"):
                manage_aliases(model_name, details.version)

            ans = input("\nManage tags now? (yes/no): ").strip().lower()
            if ans.startswith("y"):
                manage_tags(model_name, details.version)

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register MLflow model")

    parser.add_argument("--experiment", type=str, default=MLFLOW_DEFAULT_EXPERIMENT_NAME,
                        help="Experiment name (default: value from constants).")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Registered model name (default: auto-generated if not provided).")
    parser.add_argument("--run-id", type=str, default=None,
                        help="Run ID to use. If omitted, the latest FINISHED run is selected.")
    parser.add_argument("--tags", type=str, default=None,
                        help='Comma-separated tags, e.g. "stage=staging,owner=georges".')
    parser.add_argument("--filepath", type=str,
                        default="/workspace/model/model_incident_analysis_incident_duration_min.joblib",
                        help=("Artifact-relative path inside the run (file or directory). "
                              "Ignored if interactive selection is used and no path is provided."))
    parser.add_argument("--interactive", action="store_true",
                        help="Enable interactive prompts and artifact directory selection.")

    args = parser.parse_args()
    
    ## Generate default run_id if not provided
    # run_id = args.run_id or f"auto_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_register_model(
        experiment_name=args.experiment,
        model_name=args.model_name,
        run_id=args.run_id,
        tags=args.tags,
        filepath=args.filepath,
        interactive=args.interactive,
    )
    
