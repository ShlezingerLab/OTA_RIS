# Update Architecture Documentation

Use this command when asked to refresh architecture documentation from relevant Python scripts.

1. Inspect the relevant Python files first. Start with files named by the user, then check nearby entry points such as `teacher.py`, `teacher_train.py`, `training.py`, `test_demo.py`, `teacher_experiments.py`, `students.py`, `channels.py`, `flow.py`, or `CLI_interface.py` only when they are relevant.
2. Compare what the code actually does with `.cursor/rules/architecture.mdc` and the Markdown files in `.cursor/docs/architecture/`.
3. Propose or apply documentation updates that are grounded in the inspected code. Do not invent architecture, intended workflows, or future plans that are not present in the scripts or explicitly provided by the user.
4. Keep `.cursor/rules/architecture.mdc` concise and focused on stable guidance for agents. Put longer explanations in `.cursor/docs/architecture/`.
5. Preserve useful existing documentation. Replace stale claims directly instead of layering contradictory notes.
6. Keep changes simple and scoped to architecture documentation unless the user explicitly asks for code changes.

When finished, summarize which files were inspected and which documentation files changed.
