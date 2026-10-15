---
name: tcr-workbench-setup
description: >-
  Install and verify a local TCR-Workbench environment so a bench scientist can score TCR-pMHC
  data. Use when the user wants to install TCR-Workbench, set it up on their laptop or
  workstation, choose where model weights come from (HuggingFace or the shared registry), choose
  CPU / Apple Silicon / NVIDIA GPU or a model size, or diagnose why setup, the weights, or the model
  won't run. Triggers: "install tcr-workbench", "set it up", "run on my laptop / Mac / GPU",
  "pull weights from HuggingFace", "configure the registry", "which model for my hardware", "doctor
  says it's not set up", "decodertcr not installed / can't reach HuggingFace / registry not
  configured", environment/import/auth failures. NOT for analysis (see the tcr-workbench skill) or
  raw FASTQ assembly.
---

# TCR-Workbench: install and set up

Goal: get a working local environment that can score TCR–pMHC data, then verify it — with the
least friction for someone who does not live in a terminal. Setup, not analysis, is where these
users get stuck; your job is to pick the right options, install `DecoderTCR`, choose the
weight source, run the launcher commands, and translate any failure into a fix.

## Scope and prerequisites

This skill covers **agent-ready → scoring-ready**. It assumes the user already has:

- a local machine with **Python 3.9+** (the launcher needs 3.9+; setup then provisions its own
  Python 3.12 in a managed environment — `DecoderTCR` is Python 3.12 only), and
- the TCR-Workbench checkout on disk (from `git clone` or **Code → Download ZIP** on
  `akds/tcr-workbench`), and
- **a way to obtain the weights** — either **internet access to HuggingFace** (the default; a public
  model repo needs no login, a private/gated one needs a token) **or read access to a shared model
  registry** (an internal artifact directory). Setup installs the model package; weights are then
  pulled from whichever source you choose, on first use.

Getting the checkout onto the machine, and (for the registry path) knowing the registry location,
are prerequisites **outside** this skill — if a needed one is missing, point them to the onboarding
doc rather than improvising. This skill is optional: an ordinary user can run the commands below
directly.

## Mental model

Everything runs through the repository launcher: **`python3 tcr.py`**. It creates and owns a
managed `.tcr/` environment (its own Python 3.12, model runtime, germlines). The model *code* is the
**`DecoderTCR`** package (`1.5.0`); the model *weights* come from one of two sources:

- **HuggingFace (default when no registry is configured).** Setup records the Hub registry mirror it
  will use (default `hf://Decoder-CHI/DecoderTCR-staging/registry`); weights are resolved by exact
  model ID from that mirror and verified against the registry manifest (sha256/size) on first use (on
  `load`/`score`). No HPC mount needed — this is the path a laptop or a fresh sandbox uses.
- **Shared registry (internal).** Setup points the package at an existing registry root; weights are
  fetched and verified from it on first use. Use this when the user has registry access and prefers
  it. Works on every device, including Apple Silicon.

Facts that shape everything:

- **Setup installs the model package and records the weight source; it does not fetch a checkpoint
  at setup.** Weights download/verify on **first use**, not during `setup`.
- **`doctor` is read-only** — it checks that the chosen source is *reachable* (HF repo resolves, or
  registry resolves) without downloading weights, so you may run it freely.
- **HuggingFace downloads persist inside `.tcr/`** (the launcher sets `HF_HUB_CACHE` under the model
  environment), so an ephemeral sandbox reuses cached weights instead of re-downloading each session.
- **Weight source works on every device.** CPU/GPU load the safetensors directly; Apple Silicon
  (MLX) converts its bundle from the resolved artifact — a local mirror or the hf:// Hub mirror —
  and parity-checks it against the CPU reference before use.
- **Don't move the checkout after setup** — saved paths in `.tcr/` break. Pick a permanent location
  first.

Confirm `tcr.py` exists and run commands from its directory. If the skill has been copied away from
the repo, locate the user's checkout instead of guessing a path.

## Step 1 — check the current state first

Always start read-only:

```bash
python3 tcr.py doctor          # is anything already installed? what device/model/source?
python3 tcr.py doctor --deep   # also check weight-source reachability and framework/Metal imports
```

`doctor --deep` confirms the weight source is reachable (HuggingFace repo or registry) but does
**not** download weights. If it reports a healthy install for the user's intended device/model,
**stop** — no setup needed. Hand off to the tcr-workbench (analysis) skill.

## Step 2 — choose the device

| User's machine | `--device` | Notes |
|---|---|---|
| Mac with Apple M-series chip | `apple` | MLX/Metal. Requires macOS on Apple Silicon (arm64). HuggingFace or registry. |
| Linux with a working NVIDIA driver | `gpu` | CUDA. Automatic GPU setup targets Linux only. |
| Anything else — Intel Mac, Windows, no GPU, unsure | `cpu` | Always works; slower on large models. |
| Reference matching only, no model needed | `--core-only` | Installs the core; **no** model/framework/germline install. |

Guardrails the tool enforces (state the fix, don't fight them):

- `--device apple` off Apple Silicon → error; use `--device cpu`.
- `--device gpu` off Linux → error; on a Mac use `--device apple`.
- `--weight-source huggingface` works with `--device apple` — the MLX bundle is converted from the
  Hub mirror artifact (validated byte-for-byte against the CPU reference for 300M).

## Step 3 — choose the weight source

Most of the time you don't specify this — the default is right:

- **No registry configured → HuggingFace** (any device). This is the laptop / fresh-machine default.
  Setup announces the repo it will use.
- **A registry is configured (`--registry` or `DECODERTCR_REGISTRY`) → registry.**
- **Apple Silicon → HuggingFace when no registry is configured** (same default as CPU/GPU); a
  configured local registry stays the default. Apple converts its MLX bundle from either source.

Set it explicitly only when overriding the default:

| Situation | Flag |
|---|---|
| Force the HuggingFace Hub mirror | `--weight-source huggingface` |
| Force the internal registry | `--weight-source registry --registry /path/to/registry` |
| Use a non-default / private HF repository | `--weight-source huggingface --hf-repo ORG/REPO` |

**HuggingFace authentication:** a public model repo needs none. For a **private or gated** repo,
export a read token first — `export HF_TOKEN=hf_…` (or run `hf auth login` once). If the token is
missing, `doctor`/first scoring fails with a clear "not reachable / check HF_TOKEN" message; never
work around it silently.

## Step 4 — choose the model

The CLI names below select DecoderTCR variants. Weights are fetched and verified from the chosen
source on first use, not downloaded per machine at setup.

| Model | Use it when | Never |
|---|---|---|
| `esmc-300m` (default) | Laptops / CPU / Apple; the safe default | — |
| `esmc-600m` | GPU or Apple; stronger on several benchmarks | — |
| `esmc-6b` | Only a large-memory NVIDIA GPU | never on a laptop or CPU |

These `esmc-*` names map to the released **`decodertcr@1.5.0`** (300M / 600M / 6B) checkpoints, which
use the **V2** sequence convention. This is a model upgrade: scores and benchmarks can differ from
the earlier V0.3 (V1) models. Don't present old numbers as current.

All three members (300M / 600M / 6B) live in the default staging Hub registry mirror, so any of them
can use the HuggingFace path (`--weight-source huggingface`). Supply `--hf-repo ORG/REPO` only to
override the default Hub repository. The ESM2 aliases (`esm2-650m` / `esm2-3b`) are **out of scope in
this build** — do not offer them. Default is `esmc-300m`.

## Step 5 — run setup

Setup installs dependencies and `DecoderTCR==1.5.0` into the managed environment, then
records the weight source. There is no large download during setup — weights arrive on first use.

```bash
# Laptop / CPU, default 300M, weights from HuggingFace (the default with no registry):
python3 tcr.py setup --device cpu

# Same, stated explicitly:
python3 tcr.py setup --device cpu --weight-source huggingface

# Private/gated HF repo — authenticate first:
export HF_TOKEN=hf_xxx
python3 tcr.py setup --device cpu --weight-source huggingface

# Apple Silicon, default 300M, weights from HuggingFace:
python3 tcr.py setup --device apple

# Internal registry instead (any device):
python3 tcr.py setup --device cpu   --registry /path/to/decodertcr-registry
python3 tcr.py setup --device apple --registry /path/to/decodertcr-registry

# Reference matching only, no model runtime:
python3 tcr.py setup --core-only

# Other model size, or mouse germlines:
python3 tcr.py setup --device gpu --model esmc-600m --registry /path/to/decodertcr-registry
python3 tcr.py setup --device cpu --species mouse
```

Weights are fetched and verified on **first use** (on `load`/`score`), not at setup. If that first
use stops with an **estimated-memory** message, the model is too big for the machine — choose a
smaller model or better hardware. `--allow-memory-risk` overrides that guard but can exhaust memory;
**do not add it silently** — only with explicit user consent.

## Step 6 — verify

```bash
python3 tcr.py doctor --deep
```

Expect it to report the installed device, model, precision, and a reachable weight source — either
`HuggingFace source OK: <repo> <commit> …` or a reachable registry — plus healthy framework imports.
If it passes, setup is done; weights are pulled on first scoring.

## Step 7 — first run / hand-off

```bash
python3 tcr.py example --out demo   # writes synthetic inputs and prints the exact next command
```

The first real scoring on the HuggingFace path downloads ~1.3 GB (300M) once, into the `.tcr/`
cache. Follow the printed command, then hand off to the **tcr-workbench** skill for real scoring and
interpretation.

## HuggingFace access and caching

- **Public repo:** no login. **Private/gated repo:** `export HF_TOKEN=hf_…` (a read token with
  access) or `hf auth login` once. The launcher passes the token through and also honours a cached
  login.
- **Where weights live:** the launcher sets `HF_HUB_CACHE` under `.tcr/envs/model/hf-cache`, so
  downloads persist with the project. If the environment is ephemeral (a sandbox that resets),
  keeping `.tcr/` avoids re-downloading; otherwise first scoring re-fetches the weights.
- **Provenance:** each run records `weight_source`, `hf_repo`, the resolved repo-head commit and the
  catalog/manifest/artifact sha256 in the output manifest — use these to confirm which weights
  produced a result. The Hub registry always pins the current repo head commit, so there is no
  manual revision flag.

## Registry configuration and access (internal)

When using `--weight-source registry`, the model package must know where the
registry is and be able to reach it. `setup --registry <root>` does this; alternatives:

- the env var `DECODERTCR_REGISTRY=/path/to/decodertcr-registry`, or
- `decodertcr configure --registry /path/to/decodertcr-registry`, or
- `~/.config/decodertcr-public/config.json`.

Then list the inventory to confirm access:

```bash
decodertcr models          # active releases
decodertcr models --all    # also archived / blocked releases
```

If a release appears only under `--all` as archived or blocked, it is not usable as-is — choose an
active release (`decodertcr@1.5.0`). `DecoderTCR==1.5.0` resolves the released
`decodertcr@1.5.0` checkpoints; if a release needs a newer package, upgrade the package rather than
editing metadata. Weights are verified when first fetched, so a partial or wrong artifact is
rejected; scoring runs offline once the artifact is cached locally.

## Behind a corporate / TLS-inspecting proxy (e.g. CZ Biohub)

On a network that inspects TLS (Umbrella/OpenDNS + Palo Alto), the first weight fetch from
HuggingFace can fail even though the network is fine. Exported environment variables **are inherited
by the launcher and its workers**, so set the fix and re-run. The fetch happens on first `load`/`score`
(or when preparing an Apple bundle), not at `setup`, so these apply to the first scoring run too:

- **Xet CDN 403 / connection reset on the weights download.** HuggingFace serves large files over its
  Xet content CDN, which does not survive a TLS-inspecting proxy (and 403s on some private repos
  regardless). Disable it:

  ```bash
  export HF_HUB_DISABLE_XET=1
  ```

  This is the most common fix for the `Decoder-CHI/DecoderTCR-staging` mirror; the download then falls
  back to a plain HTTPS request.

- **`SSL: CERTIFICATE_VERIFY_FAILED`** — Python's certificate bundle doesn't trust the proxy's MITM
  root CA (even though `curl` does, via the macOS keychain). Point Python at the system trust store;
  exported variables reach `huggingface_hub`. On macOS:

  ```bash
  security find-certificate -a -p /System/Library/Keychains/SystemRootCertificates.keychain >  /tmp/macos-ca-bundle.pem
  security find-certificate -a -p /Library/Keychains/System.keychain                        >> /tmp/macos-ca-bundle.pem

  SSL_CERT_FILE=/tmp/macos-ca-bundle.pem REQUESTS_CA_BUNDLE=/tmp/macos-ca-bundle.pem \
  HF_HUB_DISABLE_XET=1 python3 tcr.py setup --device apple
  ```

- **Download stalls / times out** — if the network uses an explicit proxy, set `HTTPS_PROXY`/
  `HTTP_PROXY` before running. The same `HF_HUB_DISABLE_XET=1` applies to germline tooling and any
  other `huggingface_hub` transfer.

Sanity-check the network outside Python first: `hf auth whoami` (confirms token + connectivity) and
`curl -sSI https://huggingface.co`. If those work but the Python fetch still fails, it's the CA/Xet
issue above, not connectivity. Only the first fetch needs the network; scoring runs offline once the
weights are cached under `.tcr/`.

## Common failures

| Symptom | Cause | Fix |
|---|---|---|
| "The launcher needs Python 3.9 or newer" | system Python too old | install Python 3.9+; setup then provisions 3.12 itself |
| "Apple setup requires macOS on Apple Silicon (arm64)" | `--device apple` on non-Apple-Silicon | use `--device cpu` |
| "Automatic NVIDIA GPU setup targets Linux" | `--device gpu` off Linux | `--device apple` (Mac) or `--device cpu` |
| "HuggingFace weights are not reachable … check HF_TOKEN" | no network, wrong repo id, or missing token for a private/gated repo | fix network / repo id; `export HF_TOKEN=hf_…` or `hf auth login` |
| Weight fetch fails with Xet 403 / `SSL: CERTIFICATE_VERIFY_FAILED` on first scoring | TLS-inspecting proxy / Xet CDN (e.g. CZ Biohub) | `export HF_HUB_DISABLE_XET=1` (+ the CA-bundle fix) — see "Behind a corporate / TLS-inspecting proxy" |
| "HuggingFace repository must be OWNER/NAME …" | malformed `--hf-repo` value | pass a valid `OWNER/NAME`, e.g. `--hf-repo some-org/some-repo` |
| "DecoderTCR registry is not available: No local registry is configured…" | `--weight-source registry` with no registry configured | `decodertcr configure --registry <root>` / set `DECODERTCR_PUBLIC_REGISTRY` / pass `--registry <root>`, or use `--weight-source huggingface` |
| "Setup stopped … estimated memory exceeds the available budget" | model too big for RAM/VRAM | smaller model or better hardware; `--allow-memory-risk` only with consent |
| `ModuleNotFoundError: DecoderTCR` / "decodertcr is not installed" | model package missing from the environment | re-run full `setup` for the device |
| "DecoderTCR <ver> is too old; install >= 1.5.0" | model package below the required version | upgrade the model package |
| release shows only under `decodertcr models --all` as archived/blocked | that release is not usable as-is | choose an active release (`decodertcr@1.5.0`) |
| "Setup is needed" / `ModuleNotFoundError` at analysis time | not set up (or core-only, no model) | run `setup` for the device |
| `doctor` shows no model after `--core-only` | core-only intentionally skips the model runtime | re-run `setup` for the device |

## Caveats to hold to

- **Weights are fetched and verified on first use, never at setup; `doctor` only checks
  reachability.** HuggingFace is the default source on every device (including Apple Silicon); the
  registry is the internal alternative.
- **Don't switch weight source, model, device or precision silently.** The default (registry if
  configured, else HuggingFace) is announced; state it, and only override on explicit request.
- **HuggingFace auth:** public repos need none; private/gated need `HF_TOKEN` / `hf auth login` —
  surface the auth error, don't work around it.
- **Model upgrade:** the `esmc-*` names resolve `decodertcr@1.5.0` (V2 sequence convention). Scores
  can differ from the earlier V0.3 (V1) models; don't compare across the two.
- **ESM2 aliases are out of scope in this build** — don't offer `esm2-650m` / `esm2-3b`.
- **Mouse still needs `--species mouse`** on setup (installs mouse germlines) and on each analysis.
- **Precision:** FP32 is the default and required for CPU/CUDA and for 600M/6B. Apple
  `--precision float16` is an approximate option for **300M only**; don't apply it elsewhere.
- **Never `--allow-memory-risk` silently** — it can OOM the machine.
- Installing this skill does not authorize uploading the user's biological data, installing globally,
  or moving/publishing the checkout. Respect the authorization already given for the task.
