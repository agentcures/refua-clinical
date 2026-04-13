"""HTML reporting utilities for refua-clinical artifacts."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any


def render_workup_html(
    *,
    run_payload: dict[str, Any],
    protocol_payload: dict[str, Any] | None = None,
    optimization_payload: dict[str, Any] | None = None,
    voi_payload: dict[str, Any] | None = None,
    advice_payload: dict[str, Any] | None = None,
    transportability_payload: dict[str, Any] | None = None,
) -> str:
    summary = _mapping(run_payload.get("summary"))
    protocol = (
        _mapping(protocol_payload.get("protocol"))
        if isinstance(protocol_payload, dict)
        else {}
    )
    optimization = _mapping(optimization_payload) if isinstance(optimization_payload, dict) else {}
    voi = _mapping(voi_payload) if isinstance(voi_payload, dict) else {}
    advice = _mapping(advice_payload) if isinstance(advice_payload, dict) else {}
    transportability = (
        _mapping(transportability_payload) if isinstance(transportability_payload, dict) else {}
    )

    cards = [
        _card("Power", _float_text(summary.get("power"))),
        _card("Effect", _float_text(summary.get("mean_effect"))),
        _card("Safety", _float_text(summary.get("safety_event_rate"))),
        _card("Expected N", _float_text(summary.get("expected_sample_size"), digits=1)),
    ]
    if summary.get("event_rate") is not None:
        cards.append(_card("Event Rate", _float_text(summary.get("event_rate"))))

    sections = [
        """
        <section class="hero">
          <div>
            <p class="eyebrow">refua-clinical</p>
            <h1>Clinical Design Report</h1>
            <p class="lede">
              Simulation, design optimization, information value, and advice in one artifact.
            </p>
          </div>
          <div class="cards">{cards}</div>
        </section>
        """
        .format(cards="".join(cards)),
        _json_section("Run Summary", summary),
    ]

    if protocol:
        sections.append(_protocol_section(protocol))
    if optimization:
        sections.append(_json_section("Optimization", optimization.get("best_candidate")))
    if voi:
        sections.append(_json_section("VOI", voi.get("best_scenario")))
    if advice:
        sections.append(_advice_section(advice))
    if transportability:
        sections.append(_json_section("Transportability", transportability))

    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>refua-clinical report</title>
  <style>
    :root {
      --bg: #f5f1e8;
      --panel: rgba(255, 252, 246, 0.88);
      --ink: #1e2a2f;
      --muted: #617277;
      --line: rgba(30, 42, 47, 0.12);
      --accent: #b74f2f;
      --accent-2: #1f6a70;
      --shadow: 0 18px 60px rgba(30, 42, 47, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      background:
        radial-gradient(circle at top left, rgba(183, 79, 47, 0.10), transparent 35%),
        radial-gradient(circle at top right, rgba(31, 106, 112, 0.12), transparent 30%),
        linear-gradient(180deg, #f7f3eb 0%, var(--bg) 100%);
      color: var(--ink);
    }
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 18px 56px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 24px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }
    .hero {
      display: grid;
      gap: 20px;
      align-items: start;
    }
    .eyebrow {
      margin: 0 0 10px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent-2);
      font-size: 12px;
    }
    h1, h2 {
      margin: 0 0 12px;
      line-height: 1.05;
    }
    h1 { font-size: clamp(2.2rem, 5vw, 4rem); }
    h2 { font-size: 1.5rem; }
    .lede {
      max-width: 52rem;
      margin: 0;
      color: var(--muted);
      font-size: 1.05rem;
    }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }
    .card {
      border-radius: 18px;
      padding: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.84), rgba(247,240,231,0.95));
      border: 1px solid var(--line);
    }
    .card strong {
      display: block;
      font-size: 1.7rem;
      margin-top: 6px;
      color: var(--accent);
    }
    dl.meta {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin: 0;
    }
    dt {
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    dd {
      margin: 4px 0 0;
      font-size: 1rem;
    }
    ul.recs {
      margin: 0;
      padding-left: 18px;
    }
    pre {
      margin: 0;
      padding: 14px;
      overflow: auto;
      border-radius: 16px;
      background: #fbf8f3;
      border: 1px solid var(--line);
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <main>__SECTIONS__</main>
</body>
</html>
"""
    return template.replace("__SECTIONS__", "".join(sections))


def write_workup_html(path: str | Path, *, html: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")


def _protocol_section(protocol: dict[str, Any]) -> str:
    design = _mapping(protocol.get("design"))
    endpoint = _mapping(protocol.get("endpoint"))
    stats = _mapping(protocol.get("statistical_analysis"))
    items = "".join(
        [
            _meta_item("Trial", protocol.get("trial_id")),
            _meta_item("Endpoint", endpoint.get("kind")),
            _meta_item("Planned Enrollment", design.get("planned_enrollment")),
            _meta_item("Interim Every", design.get("interim_every")),
            _meta_item("Burn-in", design.get("burn_in_n")),
            _meta_item("Success Threshold", stats.get("success_posterior_threshold")),
        ]
    )
    return f"""
    <section>
      <h2>Protocol</h2>
      <dl class="meta">
        {items}
      </dl>
    </section>
    """


def _advice_section(advice: dict[str, Any]) -> str:
    recommendations = advice.get("recommendations")
    items = []
    if isinstance(recommendations, list):
        for item in recommendations[:8]:
            if isinstance(item, dict):
                action = escape(str(item.get("action") or "Unnamed recommendation"))
                rationale = escape(str(item.get("rationale") or ""))
                items.append(f"<li><strong>{action}</strong><br>{rationale}</li>")
    rendered_items = "".join(items or ["<li>No recommendations were generated.</li>"])
    return f"""
    <section>
      <h2>Recommendations</h2>
      <ul class="recs">{rendered_items}</ul>
    </section>
    """


def _json_section(title: str, payload: Any) -> str:
    title_html = escape(title)
    payload_html = escape(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return f"""
    <section>
      <h2>{title_html}</h2>
      <pre>{payload_html}</pre>
    </section>
    """


def _card(label: str, value: str) -> str:
    return (
        f'<article class="card"><span>{escape(label)}</span>'
        f"<strong>{escape(value)}</strong></article>"
    )


def _meta_item(label: str, value: Any) -> str:
    return f"<div><dt>{escape(label)}</dt><dd>{escape(str(value))}</dd></div>"


def _float_text(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}
