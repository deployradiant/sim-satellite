# Agent Notes

See README.md for architecture.

## Defaults

- Prefer simple, clean code over abstraction.
- Do not preserve backwards compatibility unless explicitly requested.
- Prefer fast implementation and verification loops over long upfront plans.
- Prefer deterministic tests and seeded randomness.
- Normalize timestamps to epoch timestamps in UTC.
- Track source, symbol, interval, and fetch time for market data.
- Use inline images in chat when they help visually show comparisons, generated
  samples, or run results.
- Never commit API keys, exchange secrets, wallet keys, or credentials.

## Subagent Workflow

- Subagents must always work in their own git worktree, not in the main local
  checkout.
- Use subagents for isolated investigation, experiments, and code changes only
  when their worktree can absorb the mess. Merge or copy back only the specific
  changes that have been reviewed.
- Do not let subagents write into the shared checkout unless the user explicitly
  requests that exception for a read-only or non-mutating task.

## Dataset Workflow

- Keep the full OpenEarthMap download, extraction, and prepared tensor data under
  `data/openearthmap/` locally when disk space allows. This directory is
  generated state and must not be committed.
- Modal training data should be persisted in the shared Modal data volume
  mounted at `/data`, not redownloaded into throwaway scratch space for each
  run. The default data volume is `sim-satellite-openearthmap-cache`, and
  `SIM_SATELLITE_MODAL_DATA_VOLUME_NAME` can override it.
- Keep full remote training datasets, extracted archives, prepared tensors, and
  generated research/mix manifests in the Modal `/data` volume whenever they are
  reused across runs. Commit the data volume after download, extraction,
  preparation, or manifest generation.
- Keep local preparation useful for inspection and local debugging with
  `uv run sim-satellite openearthmap`, but do not require the local checkout to
  contain the full dataset before starting Modal training.
- Reuse prepared tensors between runs when the preparation code, class mapping,
  image size assumptions, and tensor format have not changed. Use
  `--force-prepared` after changing those assumptions.
- Training should require prepared data manifests and fail clearly when prepared
  data is unavailable, instead of falling back to raw local PIL/image paths.

## Code Quality Review

Use the `thermo-nuclear-code-quality-review` skill as the guide for strict maintainability review.

- Push for simple, direct implementations.
- Remove complexity instead of rearranging it.
- Reject spaghetti conditionals, weak boundaries, and unearned abstractions.
- Prefer breaking changes over compatibility shims while the project is young.

## Online App Workflow

- When making web app, inference, checkpoint, or generation changes, update the
  Modal-hosted online app when possible and show the user the URL or exact
  command they can use to open it.
- Inference-only Modal web endpoints should default to the cheapest GPU that can
  serve the model. The current default is `SIM_SATELLITE_MODAL_WEB_GPU=T4`.
  Escalate to `L4` or `A10` only after a concrete memory or latency failure; do
  not spend H100-class GPUs on serving by default.
- After each training run that produces a user-facing checkpoint, make sure the
  online app is serving that checkpoint or clearly tell the user what checkpoint
  is online and what remains to update.
- If Modal cannot be updated from the current environment, report the blocker
  and provide the next command to run.
- Future parallel Modal experiments should set a unique
  `SIM_SATELLITE_MODAL_APP_NAME` to avoid web endpoint label collisions.

## Git Workflow

Agents may commit and push directly to `main` for this repository. Do not create a feature branch or pull request unless the user explicitly asks for one.
