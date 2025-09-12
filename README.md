EVM Assistant — Design Overview

Purpose
• A streamlined, data-driven app that ingests a portfolio CSV, computes Earned Value Management (EVM) metrics, assesses project risks, and presents clear, actionable summaries with an integrated Q&A experience.

Architecture at a Glance
• Core runtime: Streamlit UI + OpenAI Agents SDK (Runner/Agent).
• Models: Configurable “Agents’ Model” and “Summary Model” (e.g., Gemini/OpenAI via LiteLLM or native).
• State: Results stored in Streamlit session_state for reliable, cross-rerun display.
• No external persistence: History/trace disabled; no writes to disk or remote stores.
• Tracing disabled for stability; lightweight, human-friendly progress indicators instead.

Agents and Tools
• Orchestrator Agent
  – Role: Coordinates the pipeline in Agentic mode.
  – Flow: Ingestion → EVM Calculation → Risk Assessment → Final report merge.

• Ingestion Agent
  – Goal: Parse and validate CSV against expected headers; summarize issues/row counts.
  – Tool: parse_and_validate_csv (CSV header validation and parsing).

• EVM Calculator Agent
  – Goal: Compute portfolio metrics (CPI, SPI, PV, EV, AC, CV, SV) and totals.
  – Tools: compute_evms / compute_evms_from_csv (deterministic local compute).

• Risk Analyst Agent
  – Goal: Rank risks and propose corrective actions using computed metrics.
  – Tool: assess_risks (rule-based risk extraction from EVM results).

• Summary Agent
  – Goal: Generate a concise executive summary of portfolio health.

• Q&A Agent
  – Goal: Answer user questions strictly from the computed data (no outside knowledge).
  – Guardrail: Returns OUT_OF_SCOPE if the answer can’t be derived from data.

Modes
• Fast (local compute + 1 summary)
  – Deterministic local EVM computation, followed by a single LLM summary.
  – Minimal agent hops; optimal for speed.

• Agentic (multi-agent with handoffs)
  – Full orchestration: ingestion → compute → risk analysis → merged report.
  – Sidebar shows live progress with friendly agent names.

Key UX Features
• Sidebar “Working” tracker
  – Live step updates in Agentic mode (Ingestion, EVM Calculator, Risk Analyst).
• Results-first layout
  – Results render below inputs and persist across reruns.
  – “Scroll to results” and right-aligned “Back to top” pill link.
  – “Reset” pill link to clear session results.
• Q&A
  – Always-available sidebar Q&A with concise “Type your question…” input.
  – Bottom-of-results Q&A for deeper exploration.
  – Submitted questions are rendered prominently (“Q: …”) with answers beneath.
• Diagnostics
  – Redacted environment summary and quick model health check.
• Header-mapping UI
  – If uploaded CSV headers differ from the template, an interactive mapper rewrites the CSV to match expected headers.

How to Use
1) Choose models and risk thresholds; optionally inspect diagnostics.
2) Upload or paste your portfolio CSV (template available via “Get CSV Template”).
3) Select mode: Fast for speed, or Agentic for detailed, stepwise reasoning and risk analysis.
4) Click “Run EVM Analysis” and watch the sidebar “Working” tracker or main status panel.
5) Review the Final Report, Portfolio Heatmap, and Computed Metrics.
6) Ask questions using the sidebar Q&A or the Q&A box under results.

Data Flow
CSV → Ingestion (validate/map) → Local EVM compute → LLM Summary → Risk analysis (Agentic) → Report + Visuals + Q&A

Quality, Safety, and Privacy
• Deterministic local calculations ensure consistent metrics.
• Q&A restricted to computed data; out-of-scope guardrails enforced.
• No background tracing or history writes; nothing persisted outside the session.

Extensibility
• Add more agents (e.g., Forecasting, What-if Simulator).
• Extend tools (e.g., baselines, EV forecasting curves).
• Optional persistence (e.g., save results to JSON/DB) can be re-enabled without changing core flow.
