# Update Architecture Documentation

Use this command to refresh `.cursor/rules/architecture.mdc` from the latest code changes. The goal is to detect what recently changed in the code (via git) and fold those changes into the architecture documentation.

1. Detect the latest changes with git first:
   - `git status --short` to find uncommitted and untracked files.
   - `git diff` (working tree) and `git diff --staged` for line-level changes that are not yet committed.
   - When the working tree is clean, use `git log --oneline -n 10` and then `git show <commit>` or `git log -p` to inspect recently committed changes.
   - Focus on Python files. From the diffs, extract the added or changed classes, functions, and signatures.
2. Inspect the changed code regions directly. Do not rely on the diff alone; open the files and read the surrounding code to understand what the change actually does.
3. Compare the detected changes with `.cursor/rules/architecture.mdc` and the Markdown files in `.cursor/docs/architecture/` (if present). Reconcile: add new components, update changed signatures, and replace stale claims directly instead of layering contradictory notes.
4. Keep `.cursor/rules/architecture.mdc` concise and focused on stable guidance for agents. Put longer explanations in `.cursor/docs/architecture/`.
5. Ground every statement in inspected code. Do not invent architecture, intended workflows, or future plans that are not present in the scripts or explicitly provided by the user.
6. If the user names specific files or areas, prioritize those, but still use git to catch related changes.
7. Keep changes simple and scoped to architecture documentation unless the user explicitly asks for code changes.

When finished, summarize which changes were detected (and the git commands used to find them), which files were inspected, and which documentation files changed.
