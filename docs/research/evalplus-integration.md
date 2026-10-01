# Research: EvalPlus 0.3.1 integration (ticket #5)

AFK research note. How EvalPlus 0.3.1 runs HumanEval + MBPP (base and `+`) against an
OpenAI-compatible endpoint (`mlx_lm serve`), how it is sandboxed, and how its raw output
maps into the harness's normalized result schema (`evals/runner.py`).

All claims are traced to primary sources: the `evalplus/evalplus` GitHub repo (checked out
at tag `v0.3.1`, since that is the pin in `requirements-evalplus.txt`), its `docs/`, its
`Dockerfile`, and the `humanevalplus_release` / `mbbppplus_release` dataset release repos.

## TL;DR

- **Backend**: EvalPlus has a built-in `openai` backend that accepts any OpenAI-compatible
  server via `--base-url` — pointing it at `mlx_lm serve` on `http://127.0.0.1:8080/v1` is a
  first-class, documented configuration. No adapter code needed.
- **Generation count**: **542 generations per model row at pass@1 / temp 0** — 164 HumanEval(+)
  + 378 MBPP(+) — **not** 563 as ticket #5 guessed. The `+` variants are not extra generations;
  they reuse the same completions against extra test inputs, and `--greedy` forces
  `n_samples=1, temperature=0`.
- **Sandbox**: the official `ganler/evalplus` Docker image runs only the **evaluation**
  (`evalplus.evaluate`), not generation; generation runs on the host in `.venv-evalplus`
  against `mlx_lm serve`. The image executes untrusted generated code inside the container.
- **Output → JSON**: raw `*_eval_results.json` gives per-task `base_status` / `plus_status`;
  pass@1(base) and pass@1(plus) map directly into `score` / `subscores` of the normalized
  schema (§5).

---

## 1. Exact CLI invocation against `mlx_lm serve` (OpenAI-compatible)

### 1.1 The backend exists and takes `--base-url`

EvalPlus 0.3.1 ships an `openai` backend whose decoder wraps the `openai` Python client with a
configurable `base_url`:

> `evalplus/provider/openai.py` (v0.3.1):
> ```python
> self.client = openai.OpenAI(
>     api_key=os.getenv("OPENAI_API_KEY", "none"), base_url=base_url
> )
> ```

The provider factory wires `--backend openai` → `OpenAIChatDecoder` and passes `base_url`
through (`evalplus/provider/__init__.py`). The official README documents exactly this use
case for a local vLLM server, which is the same shape as `mlx_lm serve`:

> `README.md` (v0.3.1):
> ```shell
> # vLLM server (OpenAI-compatible)
> evalplus.evaluate --model "ise-uiuc/Magicoder-S-DS-6.7B" \
>                   --dataset [humaneval|mbpp]             \
>                   --base-url http://localhost:8000/v1    \
>                   --backend openai --greedy
> ```

`OPENAI_API_KEY` is read by the client but defaults to the literal string `"none"` when unset,
so **no real API key is needed** for `mlx_lm serve`; set a dummy to be explicit.

### 1.2 The concrete per-row command

`mlx_lm serve` exposes an OpenAI-compatible API at `http://127.0.0.1:8080/v1` (host setup per
`SETUP.md` §5; one server per matrix row, served by `.venv-core`). From the **`.venv-evalplus`**
venv, per dataset:

```bash
export OPENAI_API_KEY="dummy"   # mlx_lm serve ignores it; evalplus defaults to "none" if unset

# HumanEval + HumanEval+  (one command scores both base and +)
.venv-evalplus/bin/evalplus.evaluate \
    --model "duoneural-v3-bf16" \
    --dataset humaneval \
    --backend openai \
    --base-url http://127.0.0.1:8080/v1 \
    --greedy

# MBPP + MBPP+
.venv-evalplus/bin/evalplus.evaluate \
    --model "duoneural-v3-bf16" \
    --dataset mbpp \
    --backend openai \
    --base-url http://127.0.0.1:8080/v1 \
    --greedy
```

`--model` is sent verbatim as the `model` field of `chat.completions.create` requests
(`evalplus/gen/util/openai_request.py`), so it must equal whatever model name the running
`mlx_lm serve` reports in `/v1/models` (with `mlx_lm serve --model <path>` the served id is
the model path string; confirm per row with `curl 127.0.0.1:8080/v1/models`).

`evalplus.evaluate` with model kwargs does generation **and** evaluation in one command
(`evalplus/evaluate.py`: `evaluate(**model_kwargs)` calls `run_codegen(...)` then scores);
`evalplus.codegen` is the generation-only entry point. Both are `console_scripts` of the
`evalplus` package (`setup.cfg` `[options.entry_points]`).

### 1.3 How the `+` variants are selected

**There is no separate `--dataset humaneval+` flag.** EvalPlus *always* loads the Plus dataset:

> `evalplus/codegen.py`: `dataset == "humaneval"` → `get_human_eval_plus()`;
> `dataset == "mbpp"` → `get_mbpp_plus()`.

The Plus datasets (`HumanEvalPlus.jsonl`, `MbppPlus.jsonl`) contain, per task, both the
original tests (`base_input`) and the EvalPlus tests (`plus_input`) — confirmed by downloading
the actual release assets (`task` keys: `base_input`, `plus_input`, `contract`, `atol`, ...).
At evaluation time each solution is checked against **both**:

> `evalplus/evaluate.py::check_correctness` runs `untrusted_check(...)` on
> `problem["base_input"]` (→ `base` result) and, unless `--base-only`, on
> `problem["plus_input"]` (→ `plus` result). The console prints two scores:
> `humaneval (base tests)` and `humaneval+ (base + extra tests)`
> (same for mbpp; see also `docs/cli.md` example output).

So one generation pass yields both the base and the `+` score. `--base-only` (store_true)
restricts to base tests only (`docs/cli.md`) — **do not pass it**, since we need both.

**Chat-mode caveat.** With the `openai` backend, EvalPlus only supports **chat mode**
(`README.md`: "For other backends, only chat mode is allowed"), and
`--force-base-prompt` is explicitly rejected for it (`evalplus/provider/__init__.py`:
`assert not force_base_prompt`). The decoder wraps the prompt as a chat user message with a
fixed instruction prefix (`evalplus/provider/openai.py` +
`evalplus/codegen.py::instruction_prefix`), using `chat.completions.create` with a system
message `"You are a helpful assistant good at coding."`. `mlx_lm serve` applies the served
checkpoint's chat template — for the stock LiquidAI base row this means the model is evaluated
in chat mode (same convention as the EvalPlus leaderboard's ✨ chat rows). This is the only
supported path; there is no raw-completion mode for the openai backend.

## 2. Generation count at pass@1 / temperature 0 — 542, not 563

### 2.1 Per-dataset breakdown (verified against release assets)

| Dataset | Tasks | Source |
|---|---|---|
| HumanEval+ `v0.1.10` | **164** | counted in `HumanEvalPlus.jsonl.gz` from [humanevalplus_release@v0.1.10](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10) |
| MBPP+ `v0.2.0` | **378** | counted in `MbppPlus.jsonl.gz` from [mbppplus_release@v0.2.0](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0); release notes: "Now MBPP+ v0.2.0 has only 378 tasks (v0.1.0 has 399)" |
| **Total** | **542** | |

Pinned versions come from the code: `HUMANEVAL_PLUS_VERSION = "v0.1.10"`
(`evalplus/data/humaneval.py`), `MBPP_PLUS_VERSION = "v0.2.0"` (`evalplus/data/mbpp.py`).
The leaderboard notes corroborate: "Evaluated using HumanEval+ version 0.1.10; MBPP+
version 0.2.0" and "MBPP … use a subset (399 tasks) of hand-verified problems from
MBPP-sanitized (427)" — with v0.2.0 further trimming to 378.

**The ticket's 563 is wrong** (it likely came from 164 + 399 with the pre-v0.2.0 MBPP+).
Correct expected generation count per matrix row: **164 + 378 = 542** completions
(×4 rows = 2168 total generations for the full matrix).

### 2.2 Temperature 0 is forced by `--greedy`

`--greedy` is not merely a default; it **overrides** conflicting flags:

> `evalplus/codegen.py::run_codegen`:
> ```python
> if greedy and (temperature != 0 or bs != 1 or n_samples != 1):
>     temperature = 0.0
>     bs = 1
>     n_samples = 1
>     print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")
> ```

and codegen then calls `model.codegen(prompt, do_sample=not greedy, ...)` — so with `--greedy`,
`do_sample=False` and exactly **one sample per task** is produced
(`n_samples=1`). That is what makes pass@1 well-defined: 542 generations/row.

Note on the wire: the openai decoder still sends `temperature=0.0` (and `top_p=0.95`,
`max_tokens=768` from `DecoderBase` defaults) in the request
(`evalplus/gen/util/openai_request.py::make_request`); `mlx_lm serve` honors temp-0 greedy
decoding. The leaderboard itself ranks by "pass@1 using greedy decoding" (leaderboard notes),
matching this setup.

Also relevant: pass@k is only reported for `k ≤ n_samples` (`evalplus/evaluate.py`:
`for k in [1, 10, 100] if total.min() >= k`), so with 1 sample/task only `pass@1` appears —
exactly what the matrix wants.

## 3. Docker sandbox: invocation and what it isolates

Docker Desktop 28.4.0 is confirmed on the MBP (locked fact), so the sandbox path is the
official one — and the split is **generate on host, evaluate in Docker**.

### 3.1 The official invocation

> `README.md` (v0.3.1):
> ```bash
> # Local generation
> evalplus.codegen --model ... --dataset humaneval --backend ... --greedy
> # Code execution within Docker
> docker run --rm --pull=always -v $(pwd)/evalplus_results:/app ganler/evalplus:latest \
>            evalplus.evaluate --dataset humaneval \
>            --samples /app/humaneval/<identifier>.jsonl
> ```

Concretely for a row (after the generation step in §1.2 writes
`evalplus_results/humaneval/duoneural-v3-bf16_openai_temp_0.0.jsonl`):

```bash
docker run --rm --pull=always \
  -v $(pwd)/evalplus_results:/app \
  ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /app/humaneval/duoneural-v3-bf16_openai_temp_0.0.jsonl

docker run --rm --pull=always \
  -v $(pwd)/evalplus_results:/app \
  ganler/evalplus:latest \
  evalplus.evaluate --dataset mbpp \
  --samples /app/mbpp/duoneural-v3-bf16_openai_temp_0.0.jsonl
```

(When generation is delegated to the host, `evalplus.evaluate --dataset … --samples …` with
**no** model kwargs skips codegen entirely — `evalplus/evaluate.py` only calls `run_codegen`
when model kwargs are present.)

### 3.2 What the image is and what it isolates

The image is built from the repo's own `Dockerfile` (v0.3.1): `python:3.11-slim` + git +
`pip install ".[perf]"`, and it **pre-downloads all datasets**
(`get_human_eval_plus(); get_mbpp_plus(); get_evalperf_data()`), so evaluation inside the
container needs no network. `WORKDIR /app` matches the `-v $(pwd)/evalplus_results:/app`
mount.

Isolation properties (from the Dockerfile + `docs/cli.md` + `docs/execution.md`):

- **Untrusted generated code is `exec`'d inside the container**, not on the host —
  `docs/cli.md`: "You are strongly recommended to use a sandbox such as docker" for
  `evalplus.evaluate`; the local alternative is flagged "⚠️ regardless of the risks".
  The harness runs model-generated Python, so this is the safety boundary.
- **In-container resource limits still apply** (they're process-level, set by evalplus
  itself): per-process memory cap of `min(4GB, system max)` via
  `EVALPLUS_MAX_MEMORY_BYTES` (`evalplus/eval/__init__.py::query_maximum_memory_bytes`,
  `-1` disables), and per-test timeout
  `T = max(--min-time-limit=1s, ground-truth-time × --gt-time-limit-factor=4)`
  (`docs/execution.md`). Timeouts/OOM count as failed.
- **Parallelism**: `--parallel` defaults to half the container-visible cores
  (`evalplus/evaluate.py::n_workers = parallel or max(1, cpu_count() // 2)`).
  EvalPlus has its own `reliability_guard` (disables dangerous builtins, sets rlimits)
  inside each worker (`evalplus/eval/__init__.py`), so even within Docker it defense-in-depths.
- The EvalPlus datasets' ground truth is computed with "trusted" execution
  (`evalplus/gen/util::trusted_exec`) only for the canonical solutions; generated code goes
  through `untrusted_check`.

**No GPU needed in the container** — evaluation is pure CPU execution of small Python
functions; Docker Desktop's default VM resources are ample (echoes `SETUP.md` §5). Generation
stays on the host because the container has no Metal/MLX access.

## 4. Raw output shape and mapping to the normalized result schema

### 4.1 Files EvalPlus writes

Generation (`run_codegen`, `jsonl_fmt=True` default) writes, under `--root`
(default `evalplus_results/`):

```
evalplus_results/
  humaneval/<model-id>_openai_temp_0.0.jsonl        # sanitized solutions
  humaneval/<model-id>_openai_temp_0.0.raw.jsonl    # raw solutions (pre-sanitize)
  mbpp/<model-id>_openai_temp_0.0.jsonl
  mbpp/<model-id>_openai_temp_0.0.raw.jsonl
```

(Identifier built in `evalplus/codegen.py`:
`model.replace("/","--") + f"_{backend}_temp_{temperature}"`; each line is
`{"task_id": ..., "solution": ...}` — `codegen()` writes both files.)

Evaluation (`evalplus.evaluate --samples <file>.jsonl`) writes the results cache next to the
samples file:

> `evalplus/evaluate.py`: `result_path = samples.replace(".jsonl", "_eval_results.json")`

i.e. `evalplus_results/humaneval/<id>_openai_temp_0.0_eval_results.json`.

### 4.2 `_eval_results.json` structure

Written by `evalplus/evaluate.py`:

```json
{
  "date": "YYYY-MM-DD HH:MM",
  "hash": "<md5 of the dataset file, e.g. HumanEvalPlus-v0.1.10.jsonl>",
  "eval": {
    "HumanEval/0": [
      {
        "task_id": "HumanEval/0",
        "solution": "<sanitized code>",
        "base_status": "pass" | "fail" | "timeout",
        "plus_status": "pass" | "fail" | "timeout" | null,
        "base_fail_tests": [...],
        "plus_fail_tests": [...]
      }
    ]
  }
}
```

One array entry per completion (exactly one per task with `--greedy`). `pass@k` is **printed
to stdout, not stored in the JSON** — the adapter must either recompute it from `eval` or
capture stdout. Recompute is trivial and matches EvalPlus exactly: with n=1, c∈{0,1} per task,
`estimate_pass_at_k` reduces to the per-task pass rate (`evalplus/eval/__init__.py`), i.e.

- **base pass@1** = mean over tasks of `base_status == "pass"`
- **plus pass@1** = mean over tasks of `base_status == "pass" AND plus_status == "pass"`
  (EvalPlus defines the `+` score on the *intersection* — see `evaluate.py` `new_correct`
  computation and the printed `humaneval+ (base + extra tests)` label)

### 4.3 Concrete mapping into `{component, model, score, subscores, runtime, artifact_versions}`

Target schema from `evals/runner.py` docstring:
`{component, model, checkpoint, score, subscores, runtime_s, artifact_versions, timestamp}`.

Mapping for one matrix row (say `bf16`), from the two `*_eval_results.json` files
(humaneval + mbpp):

```python
{
  "component": "evalplus",
  "model": "bf16",                                # matrix row name
  "checkpoint": "phntmwvs/DuoNeural-v3-BF16-MLX", # what mlx_lm serve loaded
  "score": mean(humaneval_plus_pass1, mbpp_plus_pass1),
      # primary metric: plus pass@1 averaged over the two datasets (the headline
      # EvalPlus number, matching leaderboard "Average"). Both datasets scored at
      # n=1 temp=0.
  "subscores": {
      "humaneval_base_pass1":  <from humaneval _eval_results.json>,
      "humaneval_plus_pass1":  <...>,
      "mbpp_base_pass1":       <from mbpp _eval_results.json>,
      "mbpp_plus_pass1":       <...>,
      "n_tasks": {"humaneval": 164, "mbpp": 378},   # sanity: 542 generations/row
      "dataset_versions": {"humaneval_plus": "v0.1.10", "mbpp_plus": "v0.2.0"},
      "dataset_hashes":  {"humaneval": "<hash field>", "mbpp": "<hash field>"},
      "decoding": {"greedy": True, "temperature": 0.0, "n_samples": 1},
      "sandbox": "docker:ganler/evalplus:latest"
  },
  "runtime_s": <wall time of generation + docker evaluation for this row>,
  "artifact_versions": {
      "evalplus": "0.3.1",
      "mlx_lm": "<mlx_lm.__version__ from .venv-core>",
      "openai_client": "<openai.__version__ in .venv-evalplus>",
      "docker_image": "ganler/evalplus:latest@<digest>"   # record `docker inspect`
  },
  "timestamp": "<iso8601>"
}
```

Field provenance:

| Normalized field | Comes from |
|---|---|
| `component`, `model`, `checkpoint` | harness config (row matrix) |
| `score` | computed from both `*_eval_results.json` (`eval[task][0]["base_status"/"plus_status"]`) |
| `subscores.*_pass1` | per-dataset recomputation as in §4.2 (or stdout parse of the printed `pass@1` lines) |
| `subscores.dataset_hashes` | the `"hash"` field inside each `_eval_results.json` |
| `runtime_s` | measured by the adapter around codegen + docker run |
| `artifact_versions.evalplus` | `pip show evalplus` in `.venv-evalplus` (pinned `0.3.1`) |
| `artifact_versions.docker_image` | `docker image inspect ganler/evalplus:latest --format '{{.RepoDigests}}'` |

## 5. Open decisions for the implementation ticket (#6)

1. **`score` definition** — propose mean of the two plus pass@1s (leaderboard-style
   "Average"); keep all four subscores for the A/B delta table. Cheap to change.
2. **Base-row caveat** — the stock LiquidAI base has no chat training; EvalPlus's openai
   backend is chat-only, so the base row is evaluated in chat mode like every other row
   (consistent across the matrix; same as leaderboard base-model convention of noting
   chat with ✨). Flag it in the report, don't special-case it.
3. **Resume semantics** — `run_codegen(resume=True)` appends to existing sample files, so
   rerunning the same command resumes a partial generation for free; evaluation caches into
   `*_eval_results.json` (delete or pass `--i-just-wanna-run` to force re-eval).

## Sources

- EvalPlus source @ `v0.3.1`: `evalplus/codegen.py`, `evalplus/evaluate.py`,
  `evalplus/provider/openai.py`, `evalplus/provider/__init__.py`, `evalplus/provider/base.py`,
  `evalplus/data/humaneval.py`, `evalplus/data/mbpp.py`, `evalplus/data/utils.py`,
  `evalplus/eval/__init__.py`, `evalplus/gen/util/openai_request.py`, `setup.cfg`,
  `Dockerfile` — github.com/evalplus/evalplus (tag v0.3.1)
- EvalPlus docs @ v0.3.1: `docs/cli.md`, `docs/execution.md`; `README.md`
- Dataset releases: github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10
  (164 tasks, counted from `HumanEvalPlus.jsonl.gz`);
  github.com/evalplus/mbppplus_release/releases/tag/v0.2.0
  (378 tasks, counted from `MbppPlus.jsonl.gz`; release note: "only 378 tasks (v0.1.0 has 399)")
- Leaderboard corroboration: evalplus.github.io/leaderboard.html (notes 1–4: dataset
  versions, greedy pass@1 ranking, MBPP 399-task subset history)
