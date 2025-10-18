''' 
__author__ = "Mateo Villa Arias"
__contributors__ = "Georges Nassopoulos"
__copyright__ = None
__version__ = "1.0.0"
__email__ = "ingmatvillaa@gmail.com"
__status__ = "Dev"
__desc__ = "DVC data push pipeline manager"
'''

import argparse
import os
import subprocess
import sys
from typing import Optional

## Imports for "microservice" and "legacy" structures respectively
try:
	from Services.FastAPI.src.logging_utils import get_logger
except:
	from src.core.logging_utils import get_logger

## Initialize logger
logger = get_logger(__name__)

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
	"""
		Run a shell command with logging support

		Args:
			cmd (list[str]): The command to execute
			check (bool): Whether to exit on non-zero return code

		Returns:
			subprocess.CompletedProcess: The executed process
	"""
    
	logger.info("$ " + " ".join(cmd))
	p = subprocess.run(cmd, text=True, capture_output=True)
	if p.stdout:
		logger.info(p.stdout.strip())
	if p.stderr:
		logger.warning(p.stderr.strip())
	if check and p.returncode != 0:
		sys.exit(p.returncode)
	return p

def _dvc_commit_raw(force: bool = False) -> None:
	"""
		Commit raw DVC data changes, optionally forcing commit

		Args:
			force (bool): Force commit even if DVC requests confirmation
	"""
    
	if force:
		_run(["dvc", "commit", "-f", "data/raw.dvc"])
		return
	p = _run(["dvc", "commit", "data/raw.dvc"], check=False)
	if p.returncode != 0:
		msg = (p.stderr or "") + (p.stdout or "")
		if "Use `-f|--force` to force" in msg or "use `-f|--force`" in msg:
			logger.info("Retrying with --force")
			_run(["dvc", "commit", "-f", "data/raw.dvc"])
		else:
			sys.exit(p.returncode)

def run_push_raw_data_pipeline(remote: Optional[str] = None,
							   force: bool = False,
							   message: Optional[str] = None) -> None:
	"""
		Record updated raw data (data/raw.dvc) and push it to the DVC remote
		Does not touch dvc.lock

		Args:
			remote (str | None): DVC remote name
			force (bool): Force commit if required
			message (str | None): Git commit message
	"""
    
	_dvc_commit_raw(force=force)

	## Commit only dataset pointer, not lockfile
	if os.path.exists("data/raw.dvc"):
		_run(["git", "add", "data/raw.dvc"], check=False)
	commit_msg = message or "Update raw data"
	_run(["git", "commit", "-m", commit_msg], check=False)

	## Push blobs to DVC remote
	if remote:
		_run(["dvc", "push", "--remote", remote], check=False)
	else:
		_run(["dvc", "push"], check=False)

def run_push_pipeline_results_pipeline(remote: Optional[str] = None,
									   message: Optional[str] = None,
									   data_type: Optional[str] = None) -> None:
	"""
		Commit lockfile updates for pipeline outputs (except raw data)
		and push all changed DVC outputs to the remote

		Args:
			remote (str | None): DVC remote name
			message (str | None): Commit message
			data_type (str | None): Dataset type ("processed" or "train")
	"""
    
	commit_msg = message or "Update pipeline results"

	if data_type == "processed":
		_run(["dvc", "commit", "-f", "data/processed"], check=False)
	elif data_type == "train":
		_run(["dvc", "commit", "-f", "model"], check=False)
	else:  ## default: both
		_run(["dvc", "commit", "-f", "data/processed", "model"], check=False)

	if os.path.exists("dvc.lock"):
		_run(["git", "add", "dvc.lock"], check=False)
	if os.path.exists("dvc.yaml"):
		_run(["git", "add", "dvc.yaml"], check=False)
	_run(["git", "commit", "-m", commit_msg], check=False)

	if remote:
		_run(["dvc", "push", "--remote", remote], check=False)
	else:
		_run(["dvc", "push"], check=False)


def run_push_data(data_type: str, 
				  remote: Optional[str] = None, 
				  message: Optional[str] = None, 
				  force: bool = False) -> None:
	"""
		Decide which DVC pipeline to execute depending on data_type

		Args:
			data_type (str): Type of data ("raw", "processed", "train", "all")
			remote (str | None): DVC remote name
			message (str | None): Git commit message
			force (bool): Force DVC commit if needed
	"""
    
	match data_type:
		case "raw":
			run_push_raw_data_pipeline(remote=remote, force=force, message=message)
		case "processed":
			run_push_pipeline_results_pipeline(remote=remote, message=message, data_type="processed")
		case "train":
			run_push_pipeline_results_pipeline(remote=remote, message=message, data_type="train")
		case "all":
			run_push_raw_data_pipeline(remote=remote, force=force, message=message)
			run_push_pipeline_results_pipeline(remote=remote, message="Update processed results", data_type="processed")
			run_push_pipeline_results_pipeline(remote=remote, message="Update train results", data_type="train")
		case _:
			print("Unknown option")

def main():
	"""
		Command-line entry point for pushing DVC-tracked data
	"""
    
	ap = argparse.ArgumentParser(description="Push updated data to DVC remote.")
	ap.add_argument("--data", default="raw",
		choices=["raw", "processed", "train", "all"],
		help="Select which data to push"
	)
	ap.add_argument("--remote", default=None, help="DVC remote name, uses default if omitted")
	ap.add_argument("--force", action="store_true", help="Force DVC commit if required")
	ap.add_argument("--message", default=None, help="Git commit message")
	args = ap.parse_args()

	run_push_data(args.data, args.remote, args.message, args.force)

if __name__ == "__main__":
	main()