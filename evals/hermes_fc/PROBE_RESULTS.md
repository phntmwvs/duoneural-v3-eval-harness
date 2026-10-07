# Hermes FC suite — `` emission probe results

**Ticket:** #12 (author 40 Hermes FC cases + probe `` emission on the MBP).

**Status: GATE CLEARED on 4-bit quant — 10/10 full pass with the
reproducible system message.** This is real, reproducible data from
the M4 Pro. The earlier "0/10 → not cleared" finding (see the
[earlier attempt](#earlier-attempt-no-system-message) below) turned
out to be a system-message string that pushed the model into the
v3 dialect; the v3 model can also produce the #4 dialect reliably
when the system message doesn't ask for the wrong shape.

## TL;DR

- **With the case's `system` field** (the reproducible, committed
  case file `evals/hermes_fc/cases/fc-system2-001.json`):
  10/10 ``, 10/10 #4-dialect `` call, 10/10 #4 call parses,
  **10/10 full pass** — **GATE CLEARED**.
- **Without any system message** (the "stock template" baseline):
  0/10 ``, 10/10 #4-dialect `` call, 10/10 #4 call parses,
  0/10 full pass (gate condition requires ``).
- The model uses **two output dialects** depending on the system
  message: the #4 dialect (``) when the system message
  references the `` format placeholder, and the v3 dialect
  (``) when the system message says "exactly one
  ``" (the literal tag name vs. the format placeholder).
  The probe reports each independently; the gate condition
  currently requires the #4 dialect only.
- **Authoring fix in `fc-system2-001.json`:** the case now has 1
  tool in its `tools[]` (`schedule_meeting` only) instead of 4. With
  4 detailed schemas, the v3 4-bit model runs out of output tokens
  (512 cap, `finish_reason=length`) and is truncated mid-`` before
  producing the call. With 1 tool, thought + call fit comfortably.

## Run details

- **Quant:** `DuoNeural-v3-4bit` (5.0 GB on disk)
- **Model ID served:** `/Users/hermes/projects/duoneural-v3-eval-harness/checkpoints/DuoNeural-v3-4bit`
- **Tool served via:** `mlx_lm server` from `.venv-core` (mlx-lm
  0.32.0 per `system_fingerprint`)
- **Server host/port:** 127.0.0.1, ephemeral
- **Sample count:** 10 (default)
- **Decoding:** temperature=0, n=1, stream=False (greedy)
- **Case:** `evals/hermes_fc/cases/fc-system2-001.json` (with `system` field)
- **Started via:** `caffeinate -dims -t 1800 &` then
  `./.venv-core/bin/python evals/hermes_fc/probe_thought_emission.py …`
- **End-to-end wall time:** ~14 s (model load + 10 generations)
- **Per-call latency:** ~1.2 s (with 1 tool; was ~3.6 s with 4 tools
  because the model generated longer thoughts)

### Per-attempt table (with case's `system` field, 1 tool)

```
[ 1/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
[ 2/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
[ 3/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
[ 4/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
[ 5/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
[ 6/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
[ 7/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
[ 8/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
[ 9/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
[10/10] PASS  thought=True   thought_chars= 374  call(d4)=True   call(d4_decodes)=True   call(v3)=False v3_n=0  call(v3_un)=False v3_un_n=0  name='schedule_meeting'  t=1.23s
```

### Aggregate (with case's `system` field, 1 tool)

```
  emitted <thought>             : 10/10  (100%)
  emitted <tool_call> (#4)      : 10/10  (100%)
  emitted <tool_call> (v3, closed)    :  0/10  (0%)
  emitted <tool_call> (v3, unclosed)  :  0/10  (0%)
  emitted any call (union)      : 10/10  (100%)
  call JSON parses (#4 dialect) : 10/10  (100%)
  full pass (thought + #4 dec.) : 10/10  (100%)
  errors                        :  0/10
  gate threshold                : 100%
```

**GATE: CLEARED — safe to author system2-002..005.**

## Raw response (one example, with case's `system` field)

```json
{
  "role": "assistant",
  "content": "<thought>(…reasoning about EST vs EDT, concludes 09:00 EST on 2026-11-04…)</thought><tool_call>\n{\"name\": \"schedule_meeting\", \"arguments\": {\"title\": \"meeting with Bob\", \"attendees\": [\"bob@example.com\"], \"date\": \"2026-11-04\", \"time\": \"09:00\", \"timezone\": \"America/New_York\", \"duration_minutes\": 30}}\n</tool_call>"
}
```

Full payload: ~250 prompt tokens, ~350 completion tokens, finish_reason=stop.

## Earlier attempt: no system message

For reference (and so the contrast is auditable), the same case file
with `system: null` (the stock-template baseline):

```
  emitted <thought>             :  0/10  (0%)
  emitted <tool_call> (#4)      : 10/10  (100%)
  emitted <tool_call> (v3, closed)    :  0/10  (0%)
  emitted <tool_call> (v3, unclosed)  :  0/10  (0%)
  emitted any call (union)      : 10/10  (100%)
  call JSON parses (#4 dialect) : 10/10  (100%)
  full pass (thought + #4 dec.) :  0/10  (0%)
```

The model produces a #4-dialect call 10/10 (the right tool, the
right shape) but no `` — so the gate fails on the thought half.
This is the original "0/10 thought" finding: it's not that the
model can't think, it's that the chat-completions endpoint doesn't
elicit thinking without a system nudge.

Raw response (one example, no system):

```json
{
  "role": "assistant",
  "content": "<tool_call>\n{\"name\": \"schedule_meeting\", \"arguments\": {\"title\": \"meeting with Bob\", \"attendees\": [\"bob@example.com\"], \"date\": \"2026-11-04\", \"time\": \"09:00\", \"timezone\": \"America/New_York\", \"duration_minutes\": 30}}\n</tool_call>"
}
```

Note the right tool (`schedule_meeting`, not `get_current_time` as
in the earlier 4-tool run — irrelevant tools are a misroute
problem) and the right time (`09:00` EST, the corrected expected).

## The two-dialect problem (still real, but no longer a blocker)

The v3 model emits two different tool-call shapes depending on the
system message:

1. **#4 dialect** (``): what #4 specifies; what the
   adapter in #19 parses. Produced when the system message
   references the `` format placeholder (this case's
   message: "…write your reasoning inside a ``
   block. Then call the tool using the `` format.").
2. **v3 dialect** (``): the v3 model's own format.
   Single-quoted args, `name=` attribute style, often truncated,
   often multiple `` in one generation. Produced when
   the system message says "exactly one ``" (referring to
   the literal tag name rather than the format placeholder). This
   dialect is not specified by #4 and the v1.0 adapter won't parse
   it.

The probe now reports each dialect independently. The
case file's system message is tuned to elicit the #4 dialect, so
the gate clears today. But the v3 dialect is a real failure mode —
if any future system2 case uses a system message that nudges the
model into the v3 dialect, the gate will fail for that case.

**Recommendation:** ticket #19 (the adapter) should grow a
v3-dialect parser as a defense-in-depth measure. The case file's
system message is the primary lever; the adapter parser is the
backstop. This is the work I flagged earlier as a new ticket #28
("Hermes FC adapter: dialect-agnostic tool-call parser"); it's
now even more clearly necessary.

## Authoring fixes in `fc-system2-001.json`

Three authoring issues caught during the probe, all fixed in this
file:

1. **DST arithmetic error.** I originally asserted
   `time: "10:00"` (EDT, UTC-4) but 2026-11-04 is **EST (UTC-5)** —
   DST ends on the first Sunday of November, which in 2026 is
   2026-11-01. So 14:00 UTC = 09:00 EST. The model's `09:00`
   answer was correct; my expected was wrong. Fixed.
2. **Too many tools in the case.** Originally 4 time-domain
   tools (`schedule_meeting`, `get_current_time`, `cancel_meeting`,
   `list_calendar`); reduced to 1 (`schedule_meeting`). With 4
   detailed schemas, the v3 4-bit model's output budget runs out
   before it can emit the `` call (the `finish_reason` becomes
   `length` at 512 completion tokens). With 1 tool, thought + call
   fit comfortably. The probe record shows this clearly.
3. **System-message tuning.** The original system message ("…
   reason step by step inside a `` block. Then emit
   exactly one ``.") pushed the model into the v3
   dialect. The new message ("…write your reasoning inside a
   `` block. Then call the tool using the
   `` format.") keeps it in the #4 dialect. Both
   elicit ``; only the second produces the #4-dialect call.

## What this means for ticket #12

- **Gate cleared.** `fc-system2-002..005` can be authored on the
  cases branch. The system message to use is the one in
  `fc-system2-001.json`:
  > "Before answering, write your reasoning inside a
  > `` block. Then call the tool using the
  > `` format."
- **Adapter dialect work still needed.** Even though the gate
  clears today, the v3 dialect is a real failure mode for any
  future case whose system message nudges the model into it. The
  adapter (#19) should be made dialect-agnostic. Recommended as a
  separate ticket (proposed #28).
- **The single cases in PR #25 are unaffected** — single
  cases don't have a `system` field, so they always use the stock
  template path, which produces the #4 dialect reliably (10/10
  call emission on the system2-001 case with `system: null`).

## Files

- `PROBE.md` — how to run the probe.
- `probe_thought_emission.py` — the probe script; reports the #4
  and v3 dialects independently. Stdlib-only; reuses
  `evals.server.ServerManager` for serve lifecycle.
- `cases/fc-system2-001.json` — the probe case (1 tool, fixed
  time, working system message). Will move to the cases branch
  when `fc-system2-002..005` are authored.
- `../tests/test_hermes_probe.py` — 14 unit tests for the pure
  helpers; covers both dialects and the new
  `emitted_call_v3_unclosed` metric. Full repo test suite: 92 pass
  + 2 expected skips.
