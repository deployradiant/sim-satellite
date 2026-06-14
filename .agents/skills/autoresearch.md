---
name: autoresearch
description: Use when the user asks for auto research, autonomous research, parallel experiments, multi-agent experiments, training speed investigations, model-quality investigations, or hypothesis-driven improvement work where Codex should define a clear goal, spawn multiple subagents or worktrees, validate experiments, and consolidate evidence-backed progress.
disable-model-invocation: true
---

# AutoResearch

Use this skill to turn a broad improvement goal into measurable progress through bounded, parallel experiments.

## Core Idea

AutoResearch is not "think harder." It is an evidence loop:

1. Define one clear goal.
2. Establish the baseline and success metric.
3. Generate several plausible hypotheses.
4. Run independent experiments in parallel with subagents or worktrees.
5. Compare results using the same validation protocol.
6. Merge only the simplest evidence-backed wins.
7. Record what was tried, what worked, and what should run next.

## Research Contract

Before spawning agents, write a short contract:

- Goal: the concrete improvement sought.
- Metric: the number or artifact that decides success.
- Baseline: current result, command, seed, commit, data slice, and runtime.
- Budget: time, compute, cost, max agents, and max experiment length.
- Validation: exact command or run plan every experiment must use.
- Stop rule: when to merge, continue, or abandon the batch.

For training work, prefer a small shared benchmark first, such as a fixed-seed 1000-step run, plus the production-relevant metric: loss curve, images/sec, GPU utilization, checkpoint quality, validation loss, or qualitative samples.

## Hypothesis Selection

Pick 3-8 hypotheses that can be tested independently. Prefer hypotheses with:

- A direct causal story.
- A small implementation surface.
- A measurable expected signal.
- Low risk to core architecture.
- Disjoint file ownership for parallel workers.

Do not run a swarm of vague agents. Each agent needs one hypothesis, one write scope, one validation command, and one expected output.

## Dispatch Pattern

Use subagents only after the contract exists.

- Use explorers for bounded codebase questions.
- Use workers for bounded implementation experiments.
- Give workers disjoint ownership of files or modules.
- Tell workers they are not alone in the codebase and must not revert others' work.
- Prefer isolated worktrees for experiments that edit code.
- Keep the main agent on the critical path: baseline, orchestration, integration, and final decision.

Worker prompt shape:

```text
Hypothesis: <one sentence>
Goal metric: <metric>
Write scope: <files/modules>
Validation command: <command>
Budget: <time/steps>
Return: changed files, exact commands, metric table, logs/artifacts, conclusion.
Do not merge. Do not broaden scope. Do not revert unrelated changes.
```

## Validation Rules

- Use the same seed, data slice, and command across comparable experiments.
- Collect wall time, cost-sensitive metrics, and quality metrics.
- Save logs, plots, and checkpoints when they influence the decision.
- Treat failures and negative results as useful data.
- Replicate the winning experiment if the result is noisy and cheap to rerun.
- Never merge a change only because it is clever; merge it because it improves the metric without making the code worse.

## Consolidation

Create a compact comparison table:

| Hypothesis | Change | Metric Delta | Runtime/Cost | Risk | Decision |
| --- | --- | --- | --- | --- | --- |

Then:

- Merge the smallest winning change.
- Drop unclear, unvalidated, or complexity-heavy changes.
- Run normal tests and lint after integration.
- Update the README with a short research log entry whenever AutoResearch runs experiments or changes the repo's research state.
- Update configs, training plots, or other user-facing artifacts when they are part of the workflow.
- Leave a next-experiment list with the top remaining hypotheses.

README research log entries should be brief and durable:

- Date and commit or branch.
- Goal and baseline.
- Experiments run, including commands or run IDs.
- Results table or links to logs/plots.
- Decision: merged, rejected, continued, or inconclusive.
- Next best hypotheses.

## Internet Research Basis

Use current sources when the topic is fast-moving. The useful pattern from recent agent research is:

- Automated research systems work best as an explicit loop of idea generation, code, experiments, visualization, and review.
- Agent-lab workflows commonly separate literature review, experimentation, and reporting, with human checkpoints improving quality.
- Open-ended agents benefit from curriculum, reusable skill memory, environment feedback, and self-verification.
- Reflexion-style memory matters: feed failed runs, scalar metrics, and free-form critiques into the next batch.
- Tree-of-thought style search matters: compare multiple paths before committing.
- Parallel subagent search is good for broad, shallow optimization under tight budgets; deeper team-style coordination is more fragile but can help with architectural refactors.

Useful starting references:

- https://arxiv.org/abs/2408.06292
- https://arxiv.org/abs/2501.04227
- https://arxiv.org/abs/2305.16291
- https://arxiv.org/abs/2303.11366
- https://arxiv.org/abs/2305.10601
- https://arxiv.org/abs/2405.15793
- https://arxiv.org/abs/2603.29632
