import argparse
import os
import subprocess
import sys
from typing import Optional

from Services.FastAPI.src.logging_utils import get_logger

logger = get_logger(__name__)

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
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
    Record updated raw data (data/raw.dvc) and push it to the DVC remote.
    Does not touch dvc.lock.
    """
    _dvc_commit_raw(force=force)

    # Commit only the dataset pointer, not the lockfile
    if os.path.exists("data/raw.dvc"):
        _run(["git", "add", "data/raw.dvc"], check=False)
    commit_msg = message or "Update raw data"
    _run(["git", "commit", "-m", commit_msg], check=False)

    # Push blobs to DVC remote
    if remote:
        _run(["dvc", "push", "--remote", remote], check=False)
    else:
        _run(["dvc", "push"], check=False)
def run_push_pipeline_results_pipeline(remote: Optional[str] = None,
                                       message: Optional[str] = None,
                                       data_type: Optional[str] = None) -> None:
    """
    Commit lockfile updates for pipeline outputs (but not raw data) 
    and push all changed DVC outputs to the remote.
    """
    commit_msg = message or "Update pipeline results"

    # Only commit depending on type
    if data_type == "processed":
        _run(["dvc", "commit", "-f", "data/processed"], check=False)
    elif data_type == "train":
        _run(["dvc", "commit", "-f", "model"], check=False)
    else:  # default: both
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
    Decide which push pipeline to run depending on data_type.
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
    ap = argparse.ArgumentParser(description="Push updated data to DVC remote.")
    ap.add_argument("--data", default="raw",
        choices=["raw", "processed", "train", "all"],
        help="Select the data to push: raw data, processed data or all the data"
    )

    ap.add_argument("--remote", default=None, help="DVC remote name, uses default if omitted")
    ap.add_argument("--force", action="store_true", help="Force dvc commit if DVC asks for it")
    ap.add_argument("--message", default=None, help="Git commit message")
    args = ap.parse_args()

    run_push_data(args)

if __name__ == "__main__":
    main()
