# Example Python Project

This is an example Python project repository.

## Installation with uv

Make sure you have `uv` [installed](https://docs.astral.sh/uv/getting-started/installation/) on your system.
Then any `uv` command should create a fully functional local environment:

```
git clone git@github.com:szczypinski-group/example_python_project.git
cd example_python_project
uv sync
```

## Development

Default linting settings and formatting settings (using [`ruff`](https://docs.astral.sh/ruff/)) have been created within `pyproject.toml` and will
be applied if the optional dependencies have been installed.

```
git clone git@github.com:szczypinski-group/example_python_project.git
cd example_python_project
uv sync
uvx pre-commit install
```
