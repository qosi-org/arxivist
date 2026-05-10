# Contributing to ArXivist

Thank you for your interest in contributing. This document covers how to improve the skill,
fix bugs, add documentation, and submit pull requests.

---

## What can I contribute?

- **Skill improvements** — better sub-agent prompts, tighter schemas, new validation rules
- **Documentation** — clearer explanations, additional examples, corrected typos
- **Example SIRs** — pre-generated SIRs and architecture plans for well-known papers
- **Bug reports** — incorrect SIR extraction, broken generated code, schema mismatches
- **Feature requests** — new pipeline stages, additional output formats, framework support

---

## Skill file conventions

The skill lives entirely in `skill/`. The most important conventions:

**SKILL.md** is the master orchestrator. Keep it under 500 lines. It must not contain
implementation logic that belongs in a sub-agent — it orchestrates, validates, and routes.

**agents/0N_name.md** files are specialist sub-agents. Each one must have:
- A clear input contract (what it receives)
- A clear output contract (what it produces and where it writes it)
- A "What you must NOT do" section with hard boundaries
- An output checklist at the end

**schemas/*.json** are the contracts between stages. Any change to a schema must be reflected
in the sub-agent files that produce or consume that schema, and must not break the example
artifacts in `examples/`.

**templates/*.json and *.md** are the blank starting points handed to sub-agents. Keep them
complete (all fields present, even if null) and in sync with their corresponding schemas.

---

## Making changes to JSON schemas

1. Edit the schema in `skill/schemas/`.
2. Update the corresponding template in `skill/templates/` if fields were added or removed.
3. Update the sub-agent `.md` file(s) that reference the changed fields.
4. Update the example artifact in `examples/` if it is affected.
5. Run CI locally to confirm all schemas are valid JSON:

```bash
python3 -m json.tool skill/schemas/sir_schema.json > /dev/null
python3 -m json.tool skill/schemas/architecture_plan_schema.json > /dev/null
python3 -m json.tool skill/schemas/comparison_report_schema.json > /dev/null
python3 -m json.tool skill/state/pipeline_state_schema.json > /dev/null
python3 -m json.tool skill/templates/sir_template.json > /dev/null
python3 -m json.tool examples/attention-is-all-you-need/sir.json > /dev/null
python3 -m json.tool workspace/sir-registry/global_index.json > /dev/null
```

All commands must exit 0.

---

## Adding example SIRs

Example SIRs live in `examples/{paper-slug}/`. To add one:

1. Create `examples/{paper-slug}/sir.json` — a complete SIR conforming to `sir_schema.json`.
2. Create `examples/{paper-slug}/architecture_plan_summary.md` — the human-readable arch plan.
3. Add an entry to `examples/README.md`.
4. Ensure `sir.json` validates: `python3 -m json.tool examples/{paper-slug}/sir.json > /dev/null`.

Do not commit generated repository code (the Stage 4 output) — that belongs in `workspace/`
which is gitignored. Examples are reference SIRs only.

---

## Pull request checklist

- [ ] CI passes (all JSON files are valid)
- [ ] Schema changes are reflected in templates and sub-agent files
- [ ] New example SIRs validate against `sir_schema.json`
- [ ] SKILL.md remains under 500 lines
- [ ] All agent `.md` files have input contract, output contract, must-NOT-do, and checklist
- [ ] No secrets, API keys, or personal data in any committed file

---

## Reporting bugs

Open an issue with:
- The paper you were processing (arXiv ID or title)
- Which stage failed (1–6)
- The exact error or unexpected output
- The SIR version from `pipeline_state.json` if available

---

## Code of conduct

Be respectful and constructive. We are here to advance scientific reproducibility.
