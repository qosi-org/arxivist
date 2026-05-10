# FAQ

## General

**What kinds of papers does ArXivist work best with?**

Papers that describe a concrete model architecture with explicit equations, hyperparameter
tables, and benchmark results. Deep learning papers in NLP, CV, and RL tend to produce
the highest-confidence SIRs. Theoretical papers without implementation details produce
lower-confidence SIRs with more stubs.

**Does ArXivist guarantee the generated code reproduces the paper exactly?**

No. ArXivist is a best-effort system. It is honest about uncertainty — every assumption is
annotated, every ambiguity is flagged, and every low-confidence section is surfaced. The
Results Comparator (Stage 6) exists precisely to measure the gap between the generated
implementation and the paper's reported metrics.

**Can I process the same paper multiple times?**

Yes. The SIR registry is versioned. Running ArXivist on a paper that already has a registry
entry triggers an UPDATE rather than a COMMIT, incrementing the SIR version and retaining
all previous versions.

---

## The SIR

**Why does the SIR record implementation assumptions explicitly?**

Reproducibility failures in ML are often caused by details the original paper leaves
implicit — initialisation strategies, exact batch composition, tokenisation choices. By
recording these assumptions explicitly and propagating them into the generated code as
comments, ArXivist makes it easy to identify and fix the assumptions that matter.

**What should I do if a SIR section has confidence < 0.5?**

ArXivist will pause and ask you to review the flagged section. You can:
- Confirm the primary assumption and let the pipeline continue
- Provide the correct value if you know it
- Tell ArXivist to implement the affected component as a stub

**Can I edit a SIR after it is generated?**

Yes. Edit `workspace/sir-registry/{paper_id}/sir.json` directly. When you next run ArXivist
on that paper, it will detect the version change and treat the run as an UPDATE, preserving
the previous version. Then tell ArXivist to resume from Stage 3 to regenerate the
architecture plan and code with the corrected SIR.

---

## Generated code

**What framework does ArXivist use?**

PyTorch by default. If the paper explicitly mentions a different framework, ArXivist uses
that framework. The framework choice is recorded in the architecture plan with reasoning.

**What are stubs?**

Stubs are placeholder classes for components that ArXivist cannot implement faithfully
because the SIR confidence for that component is too low. Every stub has a docstring
explaining what information is missing and what needs to be filled in before training.

**The notebook fails to run. What do I do?**

Tell ArXivist: "The notebook fails with this error: {error message}". It triggers a Stage 5
repair loop with the error as additional context and regenerates the affected cells.

---

## Results Comparator

**How does the reproducibility score work?**

The score is a weighted combination of metric deviations, penalised by the average
confidence of the SIR sections that affect those metrics and by the fraction of paper
metrics that had no matching user result. A score of 0.9+ indicates an excellent
reproduction; below 0.6 indicates a significant gap requiring investigation.

**The comparator says a component is a "structural hallucination". What does that mean?**

A structural hallucination is a module in the generated code that does not correspond to
anything in the SIR's architecture graph. It could be a spurious module that Stage 4
added without basis, or a module that was correctly generated from a low-confidence SIR
section that turned out to be wrong. The hallucination report includes a suggested fix.

**Can I run Stage 6 more than once?**

Yes. Each run writes a new comparison report, overwriting the previous one. The
`verification_log.md` records the timestamp, input hash, and SIR version for each run,
so you have an audit trail of all comparisons even if the report files are overwritten.

---

## Workspace and registry

**Are generated repositories committed to git?**

No. `workspace/` contents are gitignored (only the empty scaffold files are committed).
Generated repositories and SIRs are runtime outputs that live on your local machine.

**Can I share a generated repository with others?**

Yes — just share the `workspace/paper-repos/{paper_id}/` folder. It is a self-contained
Git repository with its own README and reproducibility notes. You can also share
`workspace/sir-registry/{paper_id}/sir.json` to let others inspect or build on the SIR.

**How do I reset the workspace?**

Delete the contents of `workspace/sir-registry/` and `workspace/paper-repos/`, then
restore the bootstrapped `global_index.json`:

```bash
find workspace/sir-registry -mindepth 1 ! -name '.gitkeep' ! -name 'global_index.json' -delete
find workspace/paper-repos  -mindepth 1 ! -name '.gitkeep' -delete
echo '{"index_version":1,"total_papers":0,"papers":[]}' > workspace/sir-registry/global_index.json
```
