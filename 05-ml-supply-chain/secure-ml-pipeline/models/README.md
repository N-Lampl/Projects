# models/ (git-ignored)

`make run` writes the trained artifact + its signature here:

- `model.safetensors`      - the safe, data-only serialized model (no code-exec surface).
- `model.safetensors.sig`  - detached signature (cosign if installed, else local HMAC demo).

Neither is committed. Regenerate with `make run`.
