# Document A New Research Direction

Use this command when asked to capture a possible OTA_RIS research direction without integrating it into the main workflow yet.

Create a new Markdown file under `.cursor/docs/new_directions/` with a clear, lowercase, hyphenated name such as `ris-phase-controller.md`.

The note should include:

- Idea: the core proposal in a few sentences.
- Motivation: why it may help the OTA/RIS classification system.
- Affected scripts/classes: likely files, functions, or classes to inspect or touch later.
- Possible experiment plan: small steps that can test the idea without broad refactors.
- Risks and unknowns: reasons the direction may fail or complicate the project.
- Isolation note: why this should remain separate until experiments show it is useful.

Keep the writeup practical and code-aware. Do not edit training flows, architecture rules, or production code unless the user explicitly asks for implementation.
