# Repository hooks

Git does not run hooks out of a tracked directory on its own. Point it here once
per clone:

```bash
git config core.hooksPath .githooks
```

- `pre-commit` — refuses to commit anything under `dev/`, which holds working
  notes that stay out of the repository.

The same rule runs in CI (`.github/workflows/ci.yml`, job `hygiene`), so the
hook is the fast local signal, not the guarantee: `--no-verify` and a clone that
never ran the line above are both caught before they reach `main`.
