# Models

No trained ML models in this lab. The only "keys" are cryptographic keys
(AES-128 and textbook-RSA) generated fresh at runtime with `os.urandom` /
`secrets`. They are intentionally **not** persisted — every run regenerates
them. Never commit secret keys.
