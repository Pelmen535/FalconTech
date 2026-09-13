# Project instructions

Read [PROJECT.md](PROJECT.md) before changing code, data, models, metrics, or
the demo. It is the single project source of truth. The machine-readable
contracts in [contracts/schema.py](contracts/schema.py) take precedence when
they differ from prose. Update the status section of `PROJECT.md` after a
substantive change.

Do not use license plates as a matching feature. Do not train or evaluate on
the same vehicle identity across train and evaluation splits. Report real-data
metrics only when the dataset, split, protocol, and model version are recorded.
Never describe the synthetic smoke test as model accuracy.

The workspace uses codebase-memory-mcp. Prefer its graph tools for code
discovery when this repository has been indexed; otherwise use local search.
