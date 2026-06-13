# Agent Notes

See README.md for architecture.

## Defaults

- Prefer simple, clean code over abstraction.
- Do not preserve backwards compatibility unless explicitly requested.
- Prefer fast implementation and verification loops over long upfront plans.
- Prefer deterministic tests and seeded randomness.
- Normalize timestamps to epoch timestamps in UTC.
- Track source, symbol, interval, and fetch time for market data.
- Never commit API keys, exchange secrets, wallet keys, or credentials.

## Code Quality Review

Use the `thermo-nuclear-code-quality-review` skill as the guide for strict maintainability review.

- Push for simple, direct implementations.
- Remove complexity instead of rearranging it.
- Reject spaghetti conditionals, weak boundaries, and unearned abstractions.
- Prefer breaking changes over compatibility shims while the project is young.

## Git Workflow

Agents may commit and push directly to `main` for this repository. Do not create a feature branch or pull request unless the user explicitly asks for one.
