---
name: Bug report
about: Something went wrong during a pipeline run
title: "[Bug] Stage N — brief description"
labels: bug
assignees: ""
---

## Paper

arXiv ID or title of the paper you were processing:

## Stage that failed

- [ ] Stage 1 — Paper Parser
- [ ] Stage 2 — SIR Registry
- [ ] Stage 3 — Architecture Planner
- [ ] Stage 4 — Code Generator
- [ ] Stage 5 — Notebook Generator
- [ ] Stage 6 — Results Comparator

## What happened

A clear description of what went wrong.

## What you expected

What you expected ArXivist to produce.

## SIR version (if available)

From `workspace/sir-registry/{paper_id}/pipeline_state.json` → `sir_version` field.

## Relevant output

Paste the relevant section of Claude's output, the generated file that was wrong, or the
error message. Redact any personal information.

```
paste here
```

## Environment

- Claude model:
- How you invoked ArXivist (claude.ai / API / Claude Code):
