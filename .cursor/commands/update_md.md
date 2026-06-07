# Update Folder README (`/update_md`)

General command: refresh `README.md` in **whatever folder you are working in
now**, using only what is **relevant** from the **current conversation** and
**relevant uncommitted git changes**. Works for any subfolder in any project —
not tied to a specific area like `checkboard/`.

## What this command does

Produce or update `<folder>/README.md` so a future reader (or agent) understands:

- what that folder is for;
- what changed recently;
- how to run or use what lives there;
- what is still open.

This is **folder-local documentation**, not a project-wide bookmark (`/bookmark`)
and not architecture rules (`/update-architecture`).

## Step 1 — Pick the folder

Resolve the target directory in this order:

1. Path the user gives explicitly (e.g. `OTA_RIS/checkboard`, `src/api`).
2. Directory of the file currently open or most discussed in this chat.
3. Deepest common parent of files edited, created, or talked about in this chat.

State the chosen folder before editing. If ambiguous, ask once.

**Target file:** `<folder>/README.md`

- If `README.md` **exists** → update it (merge good sections, rewrite stale ones).
- If `README.md` **does not exist** → **create it** from scratch using the
  outline below and the gathered evidence. Do not skip or ask unless the folder
  is empty and there is nothing to document.

## Step 2 — Gather relevant evidence only

### Conversation (current chat)

Include only topics that touch **this folder** or files inside it:

- goal of the work;
- implementations and decisions;
- commands run or recommended;
- bugs, blockers, next steps the user named.

Ignore unrelated threads in the same chat.

### Git (scoped to folder)

```bash
git status --short -- <folder>
git diff -- <folder>
git diff --staged -- <folder>
```

Also include **related** paths outside the folder when the conversation clearly
depends on them (e.g. a SLURM script one level up, a shared config). Do not dump
whole-repo diffs unrelated to this folder.

### Code (folder contents)

List and read the main files in the folder. Verify README claims against actual
code: entrypoints, CLI flags, defaults, outputs, dependencies.

## Step 3 — Write or update README.md

Merge into the existing README when it is mostly right; rewrite sections that are
stale or wrong. Suggested outline (omit sections that do not apply):

```markdown
# <Title>

## Purpose

## Contents (key files)

## How it works

## Usage / commands

## Outputs

## Configuration (if any)

## Recent changes

From this conversation + relevant git diff.

## Next steps

## Notes (git status, dependencies, caveats)
```

Writing rules:

- Ground every claim in conversation evidence, git diff, or read files.
- Document **current** behavior from code, not old plans.
- Replace outdated text; do not stack contradictory paragraphs.
- Keep it concise — a working note, not a transcript.

## Step 4 — Do not

- Edit code unless the user also asked for code changes.
- Update other folders' READMEs unless the user asks.
- Touch `.cursor/plans/`, bookmarks, or architecture docs unless requested.
- Invent features or results not in chat, diff, or code.

## Step 5 — Report back

When done, briefly state:

- folder path and whether `README.md` was **created** or **updated**;
- what conversation topics and git paths were used;
- main additions or corrections;
- top open next step, if any.
