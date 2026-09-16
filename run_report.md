# Coworker-ready run report

Date: 2026-09-16

Branch: `codex/coworker-ready`, built from current `origin/main` (`ff10bb4`) with
the requested COT-only reasoning metrics integrated.

## Delivered

- All COT reasoning judge prompts, pipeline wiring, cached normalizers, and
  per-sample raw/normalized result columns; IO outputs remain excluded.
- A full enabled run config and a no-download, eight-output smoke config.
- A complete beginner setup/handoff guide at `docs/coworker_setup.md`.
- Safe import paths for an existing rclone OAuth JSON token, corrected pinned
  Drive destinations, truthful verification errors, and final-report verification.
- Smoke-adapter compatibility repair, the missing Markdown-report dependency,
  credential ignore rules, and regression tests.

## Credentials and runtime limits

No live credentials are included. Each execution machine needs its own private
`.env` and rclone configuration. The correctly named server variables are
`ABENCH_API_KEY` and `ABENCH_ADMIN_KEY`. A long Drive run needs a full OAuth JSON
token containing `refresh_token`, not a bare access-token string.

Drive read/write was verified separately on the owner's Mac. Live model testing
remains unavailable here because the bundled public tunnels were unreachable and
no local model server was listening. Offline tests use fake models and local
rclone destinations, not the owner's live token.

## Validation

**225 tests passed; 7 were skipped.** Shell syntax and changed-file Ruff checks
passed. See the final validation entry in `run_log.md`. The full repository has existing
lint findings in unrelated upstream adapters/tests; changed Python files are
checked independently. Missing external snapshot/flock prerequisites are reported
as test skips rather than successes.

## Coworker entry point

Read `docs/coworker_setup.md`, complete its private input steps, then:

```bash
abench validate configs/runs/coworker_smoke.yaml
abench doctor configs/runs/coworker_smoke.yaml
abench run configs/runs/coworker_smoke.yaml
# Only after the model, metric, workbook, and Drive checks pass:
abench run configs/runs/coworker.yaml
```
