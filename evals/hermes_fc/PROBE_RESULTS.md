# Hermes FC suite — `` emission probe results

**Ticket:** #12 (author 40 Hermes FC cases + probe `` emission on the MBP).

**Status:** **GATE NOT CLEARED on 4-bit quant — 0/10 `` emission under the
stock template.** This is real data, not a stub result. The findings force
a conversation about the System-2 scoring rule from #4 — see
[Implications for the scoring rule](#implications-for-the-scoring-rule).

## TL;DR

- **Stock template (no system override), 4-bit, 10 generations:** 0/10
  `` emission, 10/10 `` emission, 10/10 JSON parse. The model
  goes straight to the tool call with no `` block.
- **With system override** ("Before calling any tool, reason step by step
  inside a ``<thought>...</thought>`` block…"): the model
  emits a long, detailed `` block — but **switches to a different
  output dialect** that #4 does not describe (single-quoted args,
  `` style, multiple `` in one response, often truncated).
- **The model also picked the wrong tool** under the stock template:
  it called `get_current_time(timezone="America/New_York")` to
  "convert 14:00 UTC to NY time" — but `get_current_time` returns
  *now*, not a historical conversion. The model's tool selection is
  weak on this case even before we talk about ``.
- **An authoring error in `fc-system2-001.json`:** I asserted
  `time: "10:00"` (EDT, UTC-4) but on 2026-11-04 the US is on **EST
  (UTC-5)** — DST ends on the first Sunday of November, which is
  2026-11-01. 14:00 UTC = 09:00 EST. The model's `09:00` answer was
  correct; my expected was wrong. Case needs a fix.

## Run details

- **Quant:** `DuoNeural-v3-4bit` (5.0 GB on disk)
- **Model ID served:** `/Users/hermes/projects/duoneural-v3-eval-harness/checkpoints/DuoNeural-v3-4bit`
- **Tool served via:** `mlx_lm server` from `.venv-core` (mlx-lm
  0.32.0 per `system_fingerprint`)
- **Server host/port:** 127.0.0.1, ephemeral (59524 in the first run)
- **Sample count:** 10 (default)
- **Decoding:** temperature=0, n=1, stream=False (greedy)
- **Case:** `evals/hermes_fc/cases/fc-system2-001.json`
- **Started via:** `caffeinate -dims -t 1800 &` then
  `./.venv-core/bin/python evals/hermes_fc/probe_thought_emission.py …`
- **End-to-end wall time:** 13 s (model load + 10 generations)
- **Per-call latency:** 0.24 s – 2.19 s (mean ~1.1 s)

### Per-attempt table (stock template)

```
[ 1/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=2.19s
[ 2/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=1.15s
[ 3/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=1.92s
[ 4/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=1.98s
[ 5/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=1.70s
[ 6/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=0.41s
[ 7/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=0.40s
[ 8/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=0.24s
[ 9/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=0.74s
[10/10] FAIL  thought=False  call=True  decodes=True  name='get_current_time'  t=0.48s
```

### Aggregate (stock template)

```
  emitted <thought>   : 0/10  (0%)
  emitted <tool_call> : 10/10  (100%)
  call JSON parses    : 10/10  (100%)
  full pass           : 0/10  (0%)
  errors              : 0/10
```

## Raw response (one example, stock template)

```json
{
  "role": "assistant",
  "content": "<tool_call>\n{\"name\": \"get_current_time\", \"arguments\": {\"timezone\": \"America/New_York\"}}\n</tool_call>"
}
```

Full payload: 570 prompt tokens, 31 completion tokens, finish_reason=stop.

## Raw response (one example, system override)

```
<thought>Okay, so the user has a 30-minute meeting scheduled with Bob at
14:00 UTC on November 4, 2026. They want to reschedule it to the
equivalent local time in America/New_York on the same date, keeping
the duration at 30 minutes and only Bob as the attendee. The title
should be "meeting with Bob."

First, I need to figure out the local time in America/New_York for
that date. Since the user is in the Eastern Time Zone, I should check
the current time there. But wait, the user is asking to reschedule the
meeting, so maybe they want to know what time 14:00 UTC is in
America/New_York on that specific date. … (long reasoning) …

November 4 is after the daylight saving time ends, so it should be
UTC-5. So, if 14:00 UTC is the original time, subtracting 5 hours
would give 09:00 local time in America/New_York. …

I should use the schedule_meeting tool with these parameters to update
the meeting accordingly.
</thought><tool_call name="get_current_time" arguments={'timezone': 'America/New_York'}]</tool_call><tool_call name="schedule_meeting" arguments={'title': 'meeting with Bob', 'attendees': ['bob@example.com'], 'date': '2026-11-04', 'time': '09:…(truncated)
```

Note three things in this output:

1. The model **did** emit a long `` block.
2. The tool-call dialect **changed**: `<tool_call name=... arguments=…>`
   with **single-quoted args** (Python-dict style) instead of
   ``. This is **not** the #4 dialect — a strict
   `json.loads` would fail on the single quotes.
3. The response is **truncated mid-call** (cuts off at `'09:` inside
   the `schedule_meeting` time arg). The model produced multiple
   `` in one generation, the second one wasn't completed.

## Implications for the scoring rule

#4's system2 scoring rule is: *pass iff a ``<thought>...</thought>`` block
is present AND the stripped remainder passes the standard tool-call
rule.* The probe found:

- The "thought present" half is **0/10** under the stock template, and
  **3/3** under an explicit "think first" system override.
- The "stripped remainder passes the standard tool-call rule" half
  has **two dialects in play**: the #4 dialect (stock template) and
  a new dialect (system override). The adapter (ticket #19) is
  specified against the #4 dialect only.

Three paths forward — each is a #4-amendment-level conversation, not a
case-authoring decision:

1. **Drop System-2 from v1.0.** Lowest risk; we ship 35 cases across
   single + parallel + negative, document the model's `` behavior
   as a finding, and revisit when the v3 model is retrained or a
   different Hermes model is used. The harness adapter (#19) loses the
   `require_thought` codepath but the schema still has the field for
   a later re-enable.
2. **Tighten the scoring rule to "stock template, ``
   expected, ignore the system-override dialect."** No case change.
   Same effect as (1) for the scoring outcome (0/10), but keeps the
   scaffolding. The new dialect is irrelevant if we never ask the
   model to think.
3. **Add a system-override default to the System-2 cases AND teach
   the adapter to parse the new dialect.** Highest effort, but it
   matches the model's actual capability. Requires #4 to be amended
   on (a) the system-override string, (b) the parser change for the
   single-quoted-args / `<tool_call name=…` format, and (c) the
   scoring rule (the new dialect's tool calls need to be scored
   leniently because the model often truncates mid-call).

My recommendation: **(1) for v1.0** — drop System-2, ship 35 cases,
document the finding. The 0/10 result is unambiguous: the v3 model
on `mlx_lm serve`'s chat-completions endpoint does not surface
`` by default, and the v3 model only produces `` when
explicitly nudged, at which point it switches to a dialect #4 does
not cover. This is a real, reproducible finding — not a tunable.

The fallback option (2) is the cheapest variant of (1): same scoring
outcome, less work to revert later if/when a model revision fixes
this.

## Authoring bug in `fc-system2-001.json`

The expected call says `time: "10:00"` (EDT, UTC-4). On 2026-11-04 the
US is on **EST (UTC-5)**: DST ends on the first Sunday of November,
which is 2026-11-01. So the correct local time is `09:00` (EST), not
`10:00` (EDT). The model's `09:00` answer under the system override
was correct; my expected was wrong.

**This bug only matters if we keep the System-2 category.** If we drop
it (option 1 above), the case can stay as-is for the probe record, or
be deleted. If we keep it, fix the expected to `time: "09:00"`.

## Files

- `PROBE.md` — how to run the probe (unchanged).
- `probe_thought_emission.py` — the probe script (unchanged).
- `cases/fc-system2-001.json` — the probe case (has the
  09:00-vs-10:00 authoring bug noted above; will be fixed if we keep
  the category, deleted if we drop it).
- This file (`PROBE_RESULTS.md`) — the live run findings.
