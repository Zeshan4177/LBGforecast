# Emulator status — FROZEN (2026-07-11)

This repo's speculator/photulator usage predates the 2026 convention migration
(see /Users/bleisted/Dropbox/repos/speculator/plans/2026-07-10-emulator-convention-migration.md
and speculator/CONVENTIONS.md).

- It trains and loads its OWN Photulator artifacts (trained_models/model_0x0lsst_*). GATE-1 WARNING: those artifacts' behavior under load depends on the CURRENT class definition (full-module-pickle rebinding); before any reuse, determine which forward (zip vs the rejected 2026-05-14 linear edit) the weights were optimized under — see speculator/benchmarks/forward_adjudication_report.md for the adjudication method.
- Old model artifacts are preserved at
  pop-cosmos/sps_models/modelProspectorAlphaPlus/trained_models_v2legacy_20260710/.
- Do not "fix" imports here piecemeal; if this line of work is revived, migrate
  it properly through the plan above (golden-gated).
- Note: the collaborator paths /Users/fpetri/... in this repo never resolved on this machine.
