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

def main():
    ap = argparse.ArgumentParser(description="Push updated raw data to DVC remote.")
    ap.add_argument("--remote", default=None, help="DVC remote name, uses default if omitted")
    ap.add_argument("--force", action="store_true", help="Force dvc commit if DVC asks for it")
    ap.add_argument("--message", default=None, help="Git commit message")
    args = ap.parse_args()
    run_push_raw_data_pipeline(remote=args.remote, force=args.force, message=args.message)

if __name__ == "__main__":
    main()
