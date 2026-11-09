# Release validation

Local validation on November 9, 2026:

- `ruff check .`: passed.
- `ruff format --check .`: seven Python files passed.
- `mypy`: two dependency-free source files passed strict type checking.
- `pytest -q`: 31 tests passed.
- `quantization-quality-lab verify`: saved evidence is internally consistent.
- `python -m build --no-isolation`: source distribution and wheel built.
- Fresh wheel-only environment: installed-command replay from outside the checkout root passed.
- Both locally created model state dictionary hashes and byte sizes match model.json.

The first CI attempt failed during Python setup, before tests:
[run 42694137368](https://github.com/lucioduran/quantization-quality-lab/actions/runs/42694137368).
Ubuntu 26.04, selected by ubuntu-latest, had no Python 3.11.13 distribution.
The workflow now pins Ubuntu 24.04, which appears in the official Python versions
manifest for 3.11.13. This is a CI environment failure, not an inference failure.
No model measurements were rerun or changed to address it. The final passing run
and exact release commit are linked in the GitHub release notes.
