# TCR-Workbench setup evaluation contract

- These authoring-only cases evaluate the `tcr-workbench-setup` skill.
- Generated runs belong in a gitignored sibling workspace directory, not in this folder.
- A host that serves this skill as MCP resources must never publish any file under `evals/`.

## What the cases cover

| id | Behavior under test |
|---|---|
| 1 | Read-only `doctor` before setup; Apple Silicon defaults; weights fetched on first use |
| 2 | `--device apple` guardrail on an Intel Mac |
| 3 | Refusing `esmc-6b` on a CPU laptop; Windows goes through WSL |
| 4 | TLS-inspecting proxy: `HF_HUB_DISABLE_XET=1` and the CA bundle, never disabled verification |
| 5 | Linux GPU with an internal registry and `esmc-600m` |
| 6 | `--allow-memory-risk` only after explicit consent that names the risk |
| 7 | ESM2 aliases out of scope in this build |
| 8 | Private HuggingFace repo auth without writing the token into the checkout |
| 9 | `--core-only` for reference matching |
| 10 | A healthy install is not set up again; hand off to the analysis skill |

Cases 2, 3, 6, 7 and 10 assert what the agent must **not** do, so a run that produces no command
can still fail them. Each expectation names an observable command, flag or statement rather than a
tone, so a grader can cite the trace for every verdict.

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

## Hardware-dependent cases

Unlike evals for a skill that only calls remote tools, these cases exercise a local CLI, so some of
them cannot be graded on arbitrary hardware:

- Cases 1 and 2 need an Apple Silicon Mac and an Intel Mac respectively to reach a real guardrail.
- Case 5 needs a Linux machine with an NVIDIA driver and a mounted registry.
- Case 4 reproduces only behind a TLS-inspecting proxy; elsewhere grade the recommended fix rather
  than an observed failure.
- Case 6 needs a model that genuinely exceeds the machine's memory budget.

On other hardware the agent should still refuse or redirect correctly, which is what most
expectations assert. Record the machine each case ran on, and report a case that could not be
exercised as `not_run` rather than as a pass.

## Grading notes

- Treat only the agent's actual commands and messages as evidence. A summary claiming a flag was
  used is not evidence that it was.
- No case requires model weights to be downloaded. `doctor` and `doctor --deep` are read-only, and
  setup records the weight source without fetching a checkpoint, so a correct run of any case
  starts no large download.
- Keep real tokens out of runs and results. Case 8 uses a placeholder, and a recorded result must
  not contain a live `HF_TOKEN`.
- Rerun these cases after any change to `SKILL.md`, and summarize the results again.
