# Contributing

PRs are welcome, and most of them here are written with coding agents. This page is for you and
your agent; [AGENTS.md](AGENTS.md) holds the rules for the code itself.

## Start from the vision

[README → Vision & scope](README.md#vision--scope) decides what gets merged. Work outside it is
declined however well it's built: a new mode that isn't the voice, an export pipeline, tooling for
one person's workflow, a sequencer. If you're unsure whether an idea fits, open an issue or a draft
PR and ask before you build much.

Solve one problem per PR. Split unrelated fixes, and leave drive-by refactors for their own PR.

## Use a strong model, and review the work

- Use the most capable model you have, with reasoning effort to match the work's complexity and
  risk.
- Before you open the PR, have a fresh agent review the diff and fix what it finds (or say in the
  PR why you didn't). A reviewer from a different model family tends to catch more; we recommend
  one but don't require it.
- Follow [AGENTS.md](AGENTS.md): the engine's contract, the golden tests, and the checks to run for
  the files you touched.

## Show the change works

Put the evidence in the PR, not in git.

- **Sound**: before/after audio of the same phrase, or before/after spectrograms
  (`bun tools/spectrogram.ts`), and say what you hear change. Re-bless the goldens. Voice tuning
  follows [docs/hill-climb/](docs/hill-climb/README.md).
- **UI**: before/after screenshots, including at phone width; a short recording when motion or
  timing matters.
- **Bug fix**: the reproduction, kept as a test that fails before the fix and passes after.
- **Language**: a native speaker's check, and English unchanged
  ([docs/language-front-ends.md](docs/language-front-ends.md)).
- Say what you couldn't check.

## Write the PR top-down

Someone who reads only the first screen should know what changed, why, how you know it works, and
what could break. Details go below. Use the sections in the
[PR template](.github/pull_request_template.md), and keep prose short.

- **Title**: the area and what changed, in searchable words, for example
  `Play: grabbing the face before any audio starts it`.
- **Summary**: two to four sentences. Add a small visual if it makes the point faster: a diff of a
  call tree or file tree, a spectrogram, a screenshot.
- **Evidence**: before and after, as above.
- **Risk**: is this easy to undo (a two-way door) or hard to undo (one-way: a wire format, saved
  songs, a public API)? What else could it break: the goldens, the plugin, phones, other languages?
- **Details**: design choices, what you ruled out, the checks you ran with their numbers.
- **Attribution**, last line: the model and harness that wrote the code, and the reviewer's model if
  you used one, for example `Written by Claude Opus 5.5 in Claude Code; reviewed by GPT-5 in Codex.`
  Write `No agent` if you wrote it yourself.

## Nothing copyrighted

No copyrighted audio, lyrics or scores in git, and nothing derived from them. Contributions are
under the [MIT license](LICENSE).
