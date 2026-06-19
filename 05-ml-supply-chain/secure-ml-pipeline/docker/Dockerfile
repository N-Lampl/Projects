# Minimal sandbox image used ONLY to detonate the benign pickle PoC under
# `docker run --network none --read-only`. No project deps needed: the detonator
# uses stdlib pickle only.
FROM python:3.11-slim

# Run as a non-root user inside the (already read-only) container.
RUN useradd --create-home --uid 10001 sandbox
WORKDIR /sandbox
COPY docker/detonate.py /sandbox/detonate.py
USER sandbox

ENTRYPOINT []
CMD ["python", "/sandbox/detonate.py"]
