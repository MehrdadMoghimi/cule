#!/usr/bin/env python3
"""Append-only log of what has been asked, run and learned during exploration.

The point of this file is that nothing gets measured twice.  An autonomous
search over configurations is only as good as its memory: without one it
rediscovers the same fact every few hours and calls each rediscovery a result.

Three records, all JSONL, all append-only, under
`benchmark_results/exploration/`:

    hypotheses.jsonl  a question worth GPU time, its rationale, and its status.
                      Status moves open -> running -> answered|rejected|blocked.
    findings.jsonl    an answer, with the evidence that supports it and an
                      honest confidence.  A finding that contradicts an earlier
                      one supersedes it by id rather than editing it away.
    runs.jsonl        one line per GPU job launched, so a crashed or killed run
                      is distinguishable from one that was never started.

`JOURNAL.md` is rendered from all three and is the thing to read.

Usage:
    journal.py hypothesis "question" --why "..." --cost 2.5 [--tags a b]
    journal.py start   H003 --run-id protein-rescue --cmd "..."
    journal.py finish  H003 --status answered
    journal.py finding "title" --claim "..." --evidence "..." --confidence high \
                       --hypothesis H003 [--supersedes F002] [--gpu-hours 5.1]
    journal.py list [--status open] [--kind hypotheses|findings|runs]
    journal.py render
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "benchmark_results" / "exploration"
FILES = {kind: DIR / f"{kind}.jsonl" for kind in ("hypotheses", "findings", "runs")}
PREFIX = {"hypotheses": "H", "findings": "F", "runs": "R"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read(kind: str) -> list[dict]:
    path = FILES[kind]
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(kind: str, record: dict) -> dict:
    DIR.mkdir(parents=True, exist_ok=True)
    existing = read(kind)
    record = {"id": f"{PREFIX[kind]}{len(existing) + 1:03d}", "at": now(), **record}
    with FILES[kind].open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def rewrite(kind: str, records: list[dict]) -> None:
    FILES[kind].write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def update(kind: str, ident: str, **fields) -> dict:
    records = read(kind)
    for record in records:
        if record["id"] == ident:
            record.update(fields)
            record["updated"] = now()
            rewrite(kind, records)
            return record
    raise SystemExit(f"no {kind} record {ident}")


def render() -> None:
    hypotheses, findings, runs = read("hypotheses"), read("findings"), read("runs")
    by_hypothesis: dict[str, list[dict]] = {}
    for finding in findings:
        by_hypothesis.setdefault(finding.get("hypothesis", "-"), []).append(finding)

    lines = [
        "# Exploration journal",
        "",
        f"Rendered {now()}.  "
        f"{len(hypotheses)} hypotheses, {len(findings)} findings, {len(runs)} runs, "
        f"{sum(r.get('gpu_hours') or 0 for r in findings):.1f} GPU-hours accounted for.",
        "",
        "Read `findings` first; the hypothesis list is the queue, not the result.",
        "",
        "## Findings",
        "",
    ]
    superseded = {f["supersedes"] for f in findings if f.get("supersedes")}
    for finding in reversed(findings):
        mark = " *(superseded)*" if finding["id"] in superseded else ""
        lines += [f"### {finding['id']} — {finding['title']}{mark}", ""]
        lines += [f"**Claim.** {finding['claim']}", ""]
        if finding.get("evidence"):
            lines += [f"**Evidence.** {finding['evidence']}", ""]
        meta = [f"confidence: {finding.get('confidence', '?')}"]
        if finding.get("hypothesis"):
            meta.append(f"from {finding['hypothesis']}")
        if finding.get("gpu_hours"):
            meta.append(f"{finding['gpu_hours']:.1f} GPU-h")
        if finding.get("supersedes"):
            meta.append(f"supersedes {finding['supersedes']}")
        if finding.get("refs"):
            meta.append("refs: " + ", ".join(finding["refs"]))
        lines += ["<sub>" + " · ".join(meta) + "</sub>", "", f"<sub>{finding['at']}</sub>", ""]

    lines += ["## Hypotheses", "", "| id | status | est. GPU-h | question |", "|---|---|---:|---|"]
    for h in hypotheses:
        answers = ", ".join(f["id"] for f in by_hypothesis.get(h["id"], []))
        status = h.get("status", "open") + (f" → {answers}" if answers else "")
        lines.append(f"| {h['id']} | {status} | {h.get('cost') or '?'} | {h['question']} |")

    lines += ["", "## Runs", "", "| id | hypothesis | started | status | command |",
              "|---|---|---|---|---|"]
    for r in runs[-40:]:
        lines.append(f"| {r['id']} | {r.get('hypothesis', '-')} | {r['at'][:16]} | "
                     f"{r.get('status', 'launched')} | `{(r.get('cmd') or '')[:110]}` |")

    (DIR / "JOURNAL.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {DIR / 'JOURNAL.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("hypothesis")
    p.add_argument("question")
    p.add_argument("--why", default="")
    p.add_argument("--cost", type=float, default=None, help="estimated GPU-hours")
    p.add_argument("--tags", nargs="*", default=[])

    p = sub.add_parser("start")
    p.add_argument("hypothesis")
    p.add_argument("--run-id", required=True)
    p.add_argument("--cmd", required=True)

    p = sub.add_parser("finish")
    p.add_argument("ident", help="hypothesis or run id")
    p.add_argument("--status", required=True)
    p.add_argument("--note", default="")

    p = sub.add_parser("finding")
    p.add_argument("title")
    p.add_argument("--claim", required=True)
    p.add_argument("--evidence", default="")
    p.add_argument("--confidence", default="medium",
                   choices=["high", "medium", "low", "provisional"])
    p.add_argument("--hypothesis", default=None)
    p.add_argument("--supersedes", default=None)
    p.add_argument("--gpu-hours", type=float, default=None)
    p.add_argument("--refs", nargs="*", default=[])

    p = sub.add_parser("list")
    p.add_argument("--kind", default="hypotheses", choices=list(FILES))
    p.add_argument("--status", default=None)

    sub.add_parser("render")
    # NOTE: the "start" subcommand takes --cmd, so the subparser dest cannot
    # also be "cmd" -- argparse overwrites it and every start silently
    # dispatched to render instead.
    args = parser.parse_args()

    if args.action == "hypothesis":
        record = append("hypotheses", {"question": args.question, "why": args.why,
                                       "cost": args.cost, "tags": args.tags, "status": "open"})
        print(record["id"])
    elif args.action == "start":
        append("runs", {"id_hint": args.run_id, "hypothesis": args.hypothesis,
                        "cmd": args.cmd, "status": "launched"})
        update("hypotheses", args.hypothesis, status="running")
        print(f"{args.hypothesis} running")
    elif args.action == "finish":
        kind = "hypotheses" if args.ident.startswith("H") else "runs"
        update(kind, args.ident, status=args.status, note=args.note)
        print(f"{args.ident} {args.status}")
    elif args.action == "finding":
        record = append("findings", {
            "title": args.title, "claim": args.claim, "evidence": args.evidence,
            "confidence": args.confidence, "hypothesis": args.hypothesis,
            "supersedes": args.supersedes, "gpu_hours": args.gpu_hours, "refs": args.refs})
        print(record["id"])
        render()
    elif args.action == "list":
        for record in read(args.kind):
            if args.status and record.get("status") != args.status:
                continue
            head = record.get("question") or record.get("title") or record.get("cmd", "")
            print(f"{record['id']}  {record.get('status', '-'):9s}  {head[:100]}")
    else:
        render()


if __name__ == "__main__":
    main()
