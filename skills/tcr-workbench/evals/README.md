# TCR-Workbench analysis evaluation contract

- These authoring-only cases evaluate the `tcr-workbench` skill.
- Generated runs belong in a gitignored sibling workspace directory, not in this folder.
- No file under `evals/` is served. Biohub MCP enforces this in code: its pin covers only files
  outside `evals/`, its skill resource provider skips `evals/`, and its deployment check fails if a
  deployed skill lists or serves `evals/evals.json`. Another host must exclude `evals/` the same way.
- `tests/test_skill_packaging.py` checks the rest of that host's packaging rules here: served files
  are `.md`, `.txt`, `.py` or `.csv`, and this file's shape matches skill-creator's.

## What the cases cover

| id | Behavior under test |
|---|---|
| 1 | `tcr-score` selection; PLL reported as uncalibrated, never as a binding probability |
| 2 | Explicit `--species mouse`; mouse germlines come from setup, not from this skill |
| 3 | Human Decoder class II needs an alpha/beta pair; no nearby-allele substitution |
| 4 | Dual-alpha rows are alternative pairings, not independent cells or antigen evidence |
| 5 | Ranking stays within one receptor/HLA/peptide-length context; score scales never mix |
| 6 | `Unresolved` means insufficient support, not a negative binding result |
| 7 | Background draws are a descriptive comparator, not a significance test |
| 8 | Anchors, genes and cell identifiers are never repaired to make a row scoreable |
| 9 | `--allow-memory-risk` only after explicit consent; FP16 is 300M-Apple-only |

Every case asserts what the agent must **not** do, so a run that produces no command can still
fail. Each expectation names an observable command, flag, status or statement rather than a tone,
so a grader can cite the trace for every verdict.

Cases 3, 5 and 9 are deliberately double-barrelled: each carries two violations in one request,
and partial credit distinguishes an agent that caught the obvious half from one that caught both.
Case 3 additionally tests the harder half — declining the model path while still offering `screen`,
where a DRB-only restriction remains useful.

## How to run them

These follow the layout of Anthropic's
[skill-creator](https://github.com/anthropics/skills/tree/main/skills/skill-creator) skill, which
runs the cases and writes per-case results to its workspace. This directory contains no executable
runner.

A host's deterministic tests check only the file's shape: `skill_name` matching the folder name, at
least one case, unique ids, and the five keys `id`, `prompt`, `expected_output`, `files` and
`expectations` on every case. They do not run a model, so a passing test suite is not evidence that
the skill behaves correctly. Run the cases with skill-creator and summarize the results wherever the
skill is being reviewed.

Run them in a throwaway worktree, and delete this `evals/` folder there first, so the runner cannot
read the expectations it is being graded against.

## Hardware and data independence

Unlike the `tcr-workbench-setup` cases, none of these need particular hardware, installed weights or
a configured device. Every prompt describes a state the user has already reached — a finished run, a
report column, an error — so the judgement under test is reachable from the prompt alone, and
`files` is empty for all nine.

That is deliberate. These cases test workflow selection and interpretation, not execution, so they
grade identically on a laptop and on a GPU box. A case that cannot be graded without real data
does not belong here.

## Grading notes

- Treat only the agent's actual commands and messages as evidence. A summary claiming a flag was
  used is not evidence that it was.
- No case requires model weights, a download or a scored analysis. A correct run of any case starts
  no large download.
- An agent that reaches the right conclusion and then supplies the requested wording anyway has
  failed. Cases 6 and 7 fail on the delivered artifact, not on the caveat attached to it.
- Over-refusal is also a failure. Case 3 fails if the agent calls the class II data unusable rather
  than routing it to curated reference evidence.
- The flags, defaults and status names these cases assert are taken from `SKILL.md`. When the CLI
  changes, check the cases against `python3 tcr.py --help` before trusting a failure, since a stale
  expectation and a real regression look identical in the trace.
- Rerun these cases after any change to `SKILL.md`, and summarize the results again.
