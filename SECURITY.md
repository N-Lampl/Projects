# Security Policy

## Scope of this repository

This is a personal research & learning portfolio. The code here is for **education and authorized
testing only** — see [ETHICS.md](ETHICS.md) for the authorized-use rules.

## Reporting a vulnerability

If you find a security issue **in this repository's code** (e.g. a project that could harm the user
running it, a leaked secret, an unsafe default), please open a private report rather than a public
issue:

- Use GitHub's **"Report a vulnerability"** (Security ▸ Advisories) on this repo, or
- email the maintainer listed in the repo profile.

Please include reproduction steps and the affected path. I aim to acknowledge within 7 days.

## If a project surfaces a real third-party vulnerability

Some projects (e.g. the LLM red-team and ML supply-chain tracks) can, in principle, surface a genuine
flaw in third-party software or a public model. If that happens:

1. Do **not** disclose publicly first.
2. Contact the affected vendor via their published security/disclosure channel.
3. Allow reasonable time to remediate before any write-up, and redact anything that would aid abuse.

## Secrets

No credentials are committed. API keys live in a local, git-ignored `.env`. `detect-secrets` runs in
pre-commit to catch accidental leaks.
