# Create Project Bookmark

Use this command to create a comprehensive handoff note for resuming the current work later.

Create a new Markdown file in `.cursor/docs/bookmarks/` named:

`bookmark_YYYY-MM-DD.md`

If a bookmark for today already exists, append a short suffix such as:

`bookmark_YYYY-MM-DD_2.md`

The bookmark should be practical, specific, and useful to a future agent continuing this work after several days.

Before writing the bookmark:

1. Inspect the current project state:
   - `git status --short --branch`
   - `git diff`
   - `git diff --staged`
   - `git log --oneline -n 5`
2. Read any files that are directly relevant to the current conversation or changed work.
3. Do not rely only on memory. Ground the bookmark in the actual repository state.

The bookmark must include these sections:

## Goal

What we are trying to accomplish.

## Current State

Where the work stands right now, including what is already done and what is incomplete.

## Important Context

Key decisions, assumptions, constraints, or user preferences that matter for continuing later.

## Files And Areas

Relevant files, directories, commands, experiments, or docs.

## Git State

Branch name, uncommitted changes, staged changes, recent commits, and anything unusual.

## Commands Run

Important commands already run and their results, especially tests, training runs, linting, scripts, or failed attempts.

## Open Questions

Anything unresolved that the next session should clarify before continuing.

## Next Steps

The exact next 3-7 steps a future agent should take.

## Resume Prompt

A short copy-paste prompt the user can use later to continue from this bookmark.

When finished, summarize the bookmark file path and the most important next step.
