# Contributing

Use Python 3.11.13. Install requirements-dev.txt and the editable package.
Run ruff check ., ruff format --check ., mypy and pytest -q before a pull request.
Keep code, fixtures, documentation and discussion in English. Do not use em dash characters.

Do not replace measured evidence silently. Freeze changes to tasks and protocols
before new measurements, preserve unsuccessful runs, and use a separate evidence
revision. All local inference must use the shared exclusive lock described in
README.md. CI is intentionally offline and never loads model weights.

Bug reports should include the command, Python version and complete error message.
Do not submit private data. Original fixture contributions use the MIT license.
