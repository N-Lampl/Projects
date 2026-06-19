# data/ (git-ignored)

This lab has **no dataset**. The "target" is the OWASP Juice Shop container, and the offline
default path uses an in-process deterministic mock (`src/juiceshop_lab/client.py:MockJuiceShop`).

- **Target app:** [OWASP Juice Shop](https://owasp.org/www-project-juice-shop/) — a deliberately
  insecure web app for security training. License: MIT.
- **Run the real target (optional, needs docker):**
  ```bash
  docker compose up -d        # pulls bkimminich/juice-shop, binds 127.0.0.1:3000 only
  open http://localhost:3000
  ```
  The container image (~ a few hundred MB) is pulled by docker, never committed here.
- **Authorized use only:** the container is your own local, deliberately-vulnerable instance.
  See [../../../ETHICS.md](../../../ETHICS.md). Never point these scripts at a host you do not own.
