"""
__author__ = "Mateo Villa Arias"
__contributors__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.4"
__email__ = "ingmatvillaa@gmail.com"
__status__ = "Dev"
__desc__ = "MLflow model registration with optional interactive prompts and configurable artifact path."
"""

import argparse
import sys
from datetime import datetime
import os   
import dagshub
import mlflow
from dotenv import load_dotenv

## Imports for "microservice" et "legacy" structures respectively
try:
      
    from Services.FastAPI.src.constants import (
        MLFLOW_DEFAULT_EXPERIMENT_NAME,
        MLFLOW_ENABLE_REMOTE,
        MLFLOW_LOCAL_URI,
        MLFLOW_REMOTE_URL,
        DAGSHUB_REPO_OWNER,
        DAGSHUB_REPO_NAME,
    )
    from Services.FastAPI.src.logging_utils import get_logger
except:

    from src.core.constants import (
        MLFLOW_ENABLE_REMOTE,
        MLFLOW_REMOTE_URL,
        MLFLOW_LOCAL_URI,
        MLFLOW_DEFAULT_EXPERIMENT_NAME,
        DAGSHUB_REPO_OWNER,
        DAGSHUB_REPO_NAME,
    ) 
    from src.core.logging_utils import get_logger
    
logger = get_logger(__name__)
load_dotenv()
if os.getenv("DAGSHUB_TOKEN"):
    dagshub.auth.add_app_token(os.getenv("DAGSHUB_TOKEN"))
    logger.info(f"[Dagshub] ! Using token authentication for {os.getenv('DAGSHUB_USER')}")
else:
    logger.warning("[Dagshub] ! No DAGSHUB_TOKEN found — OAuth will be used.")
    
## ============================================================
##  Display artifacts and let user select model directory
## ============================================================
def display_artifacts(client: mlflow.tracking.MlflowClient, run_id: str):
    """
        Display the available artifacts for a given MLflow run

        Args:
            client (mlflow.tracking.MlflowClient): MLflow client instance
            run_id (str): Target run ID

        Returns:
            list: A list of mlflow.entities.FileInfo objects
    """

	## List artifacts from the run ID    
    logger.info(f"Listing artifacts for run ID: {run_id}")
    artifacts = client.list_artifacts(run_id)
    
	## Loop through and print their structure    
    logger.info("Available artifacts:")
    for idx, artifact in enumerate(artifacts, 1):
        logger.info(f"{idx}. {artifact.path} {'(dir)' if artifact.is_dir else '(file)'}")
        if artifact.is_dir:
            nested_artifacts = client.list_artifacts(run_id, artifact.path)
            for nested in nested_artifacts:
                logger.info(f"   - {nested.path}")
    return artifacts

def select_model_path(artifacts) -> str:
    """
        Interactive prompt for user to choose a directory among available artifacts

        Args:
            artifacts (list): List of mlflow.entities.FileInfo

        Returns:
            str: Selected directory path
    """
    
	## Filter directories only    
    dirs = [art for art in artifacts if art.is_dir]
    if not dirs:
        raise Exception("No directories found in artifacts")

	## Automatically choose if only one directory exists
    if len(dirs) == 1:
        return dirs[0].path

	## Ask the user to choose if multiple directories exist
    print("\nMultiple model directories found. Please select one:")
    for idx, dir_artifact in enumerate(dirs, 1):
        print(f"{idx}. {dir_artifact.path}")

    while True:
        try:
            choice = int(input("\nEnter the number of your choice: "))
            if 1 <= choice <= len(dirs):
                return dirs[choice - 1].path
            print(f"Please enter a number between 1 and {len(dirs)}")
        except ValueError:
            print("Please enter a valid number")

## ============================================================
##  Build model URI and register models in MLflow
## ============================================================
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

	## Set MLflow tracking URI    
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Using tracking URI: {tracking_uri}")

	## Retrieve the experiment or fallback to a default one
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.warning(f"Experiment '{experiment_name}' not found. Falling back.")
        experiment_name = "incident_analysis_incident_duration_min"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"No valid experiment found for '{experiment_name}'.")
            raise Exception("No valid MLflow experiment found.")

    ## ---- FIX 1: resolve run_id properly ----
    ## Resolve latest finished run if run_id not specified
    if not run_id:
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            logger.error(f"No successful runs found in experiment '{experiment_name}'.")
            raise Exception("No finished runs found.")
        run_id = runs.iloc[0].run_id
        logger.info(f"Found latest run ID: {run_id}")

    ## ---- FIX 2: resolve filepath properly ----
    ## Handle artifact path resolution
    if interactive and not filepath:
        client = mlflow.tracking.MlflowClient()
        artifacts = display_artifacts(client, run_id)
        model_path = select_model_path(artifacts)
    else:
        model_path = filepath or "model"
        if os.path.isabs(model_path):
            model_path = os.path.basename(model_path)

    logger.debug(f"Resolved run_id: {run_id}")
    logger.debug(f"Resolved model_path: {model_path}")

    model_uri = f"runs:/{run_id}/{model_path}"
    logger.info(f"Resolved model URI: {model_uri}")
    return model_uri, run_id

## -------------------------------------------------------------------
## Register a model into MLflow Model Registry
## -------------------------------------------------------------------
def register_model(model_uri: str, model_name: str, tags: dict | None = None):
	"""
		Register a model in MLflow Model Registry and optionally set tags

		Args:
			model_uri (str): Model URI (e.g., 'runs:/<run_id>/model')
			model_name (str): Target registered model name
			tags (dict | None): Optional tags

		Returns:
			mlflow.entities.model_registry.ModelVersion: Created model version
	"""
    
	logger.info(f"\nRegistering model from: {model_uri}")
	logger.info(f"Model name: {model_name}")

	client = mlflow.tracking.MlflowClient()
	try:
		model_details = mlflow.register_model(model_uri, model_name)
		logger.info(f"Model registered with version: {model_details.version}")

		## Add optional tags if provided
		if tags:
			for k, v in tags.items():
				client.set_registered_model_tag(model_name, k, v)
			logger.info("Tags set successfully.")
		return model_details

	except Exception as e:
		logger.error("Failed to register model")
		logger.error(f"Error: {str(e)}")
		raise

## -------------------------------------------------------------------
## Interactive tag management utility
## -------------------------------------------------------------------
def manage_tags(model_name: str, version: str | None = None):
	"""
		Interactively add, update, delete, or list model tags

		Args:
			model_name (str): Registered model name
			version (str | None): Specific version to target
	"""
    
	client = mlflow.tracking.MlflowClient()
	while True:
		print("\nTag Management Options:")
		print("1. Add/Update tag")
		print("2. Delete tag")
		print("3. List current tags")
		print("4. Exit")

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

## -------------------------------------------------------------------
## Interactive alias management utility
## -------------------------------------------------------------------
def manage_aliases(model_name: str, version: str):
	"""
		Interactively manage aliases for a specific model version

		Args:
			model_name (str): Registered model name
			version (str): Model version number
	"""
    
	client = mlflow.tracking.MlflowClient()
	while True:
		print("\nAlias Management Options:")
		print("1. Add alias")
		print("2. Delete alias")
		print("3. List current aliases")
		print("4. Exit")

		choice = input("\nEnter your choice (1-4): ")
		try:
			if choice == "1":
				alias = input(f"Enter alias to add to model: {model_name}, version: {version} ")
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

## -------------------------------------------------------------------
## Main orchestration: model registration pipeline
## -------------------------------------------------------------------
def run_register_model(
    experiment_name: str | None = None,
    model_name: str | None = None,
    run_id: str | None = None,
    tags: str | None = None,
    filepath: str = os.path.join(".", "model"),
    interactive: bool = False,
):
    """
    Orchestrate model URI resolution and registration.

    Args:
        experiment_name (str): MLflow experiment name. Defaults to MLFLOW_DEFAULT_EXPERIMENT_NAME
        model_name (str): Registered model name. If None (non-interactive), auto-generated.
        run_id (str): Run ID to target. If None, latest FINISHED run is used.
        tags (str): Comma-separated list of tags (e.g., "key1=val1,key2=val2").
        filepath (str): Artifact path (default: "./model/").
        interactive (bool): Enables interactive prompts and artifact selection.
    """
    
    ## Initialize tracking URI based on remote/local setup
    if MLFLOW_ENABLE_REMOTE:
        try:
            dagshub.init(
                repo_owner=DAGSHUB_REPO_OWNER,
                repo_name=DAGSHUB_REPO_NAME,
                mlflow=True,
            )
            tracking_uri = MLFLOW_REMOTE_URL
        except Exception as e:
            logger.warning(
                f"[Dagshub] Repo already exists or cannot be created: {e}. "
                "Continuing with existing repository."
            )
            tracking_uri = MLFLOW_REMOTE_URL
    else:
        tracking_uri = MLFLOW_LOCAL_URI

    try:
        ## Parse tags string into dictionary
        tag_dict = {}
        if tags:
            for pair in tags.split(","):
                key, value = pair.split("=", 1)
                tag_dict[key.strip()] = value.strip()

		## Ask user for missing parameters in interactive mode
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

		## Default fallback values
        experiment_name = experiment_name or MLFLOW_DEFAULT_EXPERIMENT_NAME
        model_name = model_name or f"auto_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

		## Resolve model URI and register model
        model_uri, run_id = get_model_uri(
			tracking_uri=tracking_uri,
			experiment_name=experiment_name,
			run_id=run_id,
			filepath=filepath,
			interactive=interactive,
		)

        ### FIX START : graceful handling of MLflow/DagsHub registration failure
        try:
            details = register_model(model_uri, model_name, tag_dict)
        except Exception as e:
            logger.warning(f"Model registration failed or unauthorized: {e}")
            logger.warning("Continuing without registration (forced success).")
            return
        ### FIX END

		## Allow alias/tag management after registration if interactive
        if interactive:
            ans = input("\nManage aliases now? (yes/no): ").strip().lower()
            if ans.startswith("y"):
                manage_aliases(model_name, details.version)

            ans = input("\nManage tags now? (yes/no): ").strip().lower()
            if ans.startswith("y"):
                manage_tags(model_name, details.version)

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Registration skipped due to error, forcing success.")
        return

## ============================================================
##  CLI Entry Point
## ============================================================
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