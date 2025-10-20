from __future__ import annotations

import pathlib
import subprocess


ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
DOCKERFILE_NAME = "Dockerfile"


def run_docker_build_smoke(*, context: pathlib.Path | None = None, tag: str = "llm-orch:smoke") -> subprocess.CompletedProcess[bytes]:
    build_context = context or ROOT_DIR
    dockerfile_path = build_context / DOCKERFILE_NAME
    command: tuple[str, ...] = (
        "docker",
        "build",
        "--file",
        str(dockerfile_path),
        "--tag",
        tag,
        str(build_context),
    )
    return subprocess.run(command, check=True)


def main() -> None:
    run_docker_build_smoke()


if __name__ == "__main__":
    main()
