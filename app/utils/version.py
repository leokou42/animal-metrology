from pathlib import Path
import tomllib


def _read_project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"

    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    return data["project"]["version"]


APP_VERSION = _read_project_version()
