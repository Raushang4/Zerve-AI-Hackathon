# GitHub Copilot / AI Contributor Instructions

Purpose
- This repo currently contains only a `LICENSE` file and Git metadata. Default branch: `main`.
- These instructions tell an AI coding agent how to be immediately productive given the current repository state.

When the codebase is empty or minimal
- Confirm repository state: run `git branch -a`, `git log --name-only --oneline -n 50`, and `ls -la` to enumerate history and any untracked content.
- Check remote repo and PRs: use `git remote -v`, `gh pr list --repo ${OWNER}/${REPO}` and `gh issue list` when available.
- If no source files exist, do not invent a project layout. Instead produce a short proposal (1 paragraph) describing the recommended project type (language, framework, deps), example file tree, and why.

How to gather context here
- Look for past file patterns in commit history: `git log --name-only --pretty=format:%h -n 200` and inspect blobs in commits that look like source files.
- If repository has other branches, inspect them: `git checkout <branch>` or view via `gh`/GitHub web.

Actionable first tasks for an AI agent
- Create a minimal `README.md` that states repository intent and development steps (ask the maintainer for missing domain info first).
- Propose a minimal project scaffold based on probable language (example: `python` → `pyproject.toml`, `src/`, `tests/`; `node` → `package.json`, `src/`, `test/`). Include exact commands to build/run/tests.
- Open a draft PR with the scaffold and a clear description of assumptions and requested maintainer decisions.

Conventions for contributions
- Use clear, short commit messages in imperative form. Example: `feat: add project scaffold`, `chore: add README`.
- Keep PRs small and focused. Put design decisions and open questions in the PR description.

If you cannot proceed
- Ask a human: open an issue or request clarification in the PR describing what domain, language, and runtime the project should use.

Files of immediate relevance
- `LICENSE` — repository license; respect license constraints when adding code or examples.

Questions for maintainers (include in PR description)
- What is the intended language/runtime for this project?
- Are there existing CI, style, or security policies to follow (external org rules)?

If this file should be merged rather than replaced
- Preserve any existing `.github/copilot-instructions.md` content that documents non-obvious project constraints. Merge new guidance into a concise single file and call out removed/updated sections in the PR.

Feedback request
- If anything is unclear or you prefer a specific scaffold (language/framework), indicate it in a reply and I will update the proposed `README.md` and scaffold accordingly.
