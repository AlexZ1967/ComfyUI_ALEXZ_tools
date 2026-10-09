"""Run a prepared update, optionally supporting legacy shutdown waiting."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))

from module_updates.executor import wait_and_execute


if __name__ == "__main__":
    wait_and_execute(Path(sys.argv[1]), int(sys.argv[2]), immediate="--immediate" in sys.argv[3:])
