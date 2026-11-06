from __future__ import annotations

import sys
import subprocess
from pathlib import Path
from typing import Callable

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture
def run_cli(project_root: Path) -> Callable[..., None]:
    def run(*arguments: str) -> None:
        subprocess.run([sys.executable, *arguments], cwd=project_root, check=True)

    return run
