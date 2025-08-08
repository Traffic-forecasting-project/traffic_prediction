import argparse
import os
import subprocess
import sys
from typing import Optional

from src.core.logging_utils import get_logger

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

def run_push_pipeline_results_pipeline(remote: Optional[str] = None,
                                       message: Optional[str] = None) -> None:
    """
    Commit lockfile updates and push all changed DVC outputs to the remote.
    """
    commit_msg = message or "Update pipeline results"
    if os.path.exists("dvc.lock"):
        _run(["git", "add", "dvc.lock"], check=False)
    if os.path.exists("dvc.yaml"):
        _run(["git", "add", "dvc.yaml"], check=False)
    _run(["git", "commit", "-m", commit_msg], check=False)

    if remote:
        _run(["dvc", "push", "--remote", remote], check=False)
    else:
        _run(["dvc", "push"], check=False)

def main():
    ap = argparse.ArgumentParser(description="Push pipeline results to DVC remote.")
    ap.add_argument("--remote", default=None, help="DVC remote name, uses default if omitted")
    ap.add_argument("--message", default=None, help="Git commit message")
    args = ap.parse_args()
    run_push_pipeline_results_pipeline(remote=args.remote, message=args.message)

if __name__ == "__main__":
    main()
