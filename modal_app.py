"""
Modal deployment for the ESCI Hybrid Search API.

This file does three things:
  1. Defines the container image (matches Dockerfile, but built by Modal)
  2. Defines a persistent Volume where artifacts live (one-time upload)
  3. Wraps the FastAPI app as a serverless ASGI endpoint

Usage:
  pip install modal
  modal setup                    # one-time auth (opens browser)

  # 1. Create the volume (idempotent — safe to re-run)
  modal volume create esci-artifacts

  # 2. Upload your local artifacts ONCE.
  #    Run this from your project root.
  modal volume put esci-artifacts artifacts/systemA /artifacts/systemA
  modal volume put esci-artifacts artifacts/matryoshka_models/us /artifacts/matryoshka_models/us
  modal volume put esci-artifacts data/processed /processed

  # 3. Deploy
  modal deploy modal_app.py

  # Modal returns a URL like https://<workspace>--esci-search-fastapi-app.modal.run
  # Test it:
  curl https://<workspace>--esci-search-fastapi-app.modal.run/health

Notes on cost:
  - CPU container, 16 GB RAM. Costs ~$0.05 per active hour.
  - Modal free tier ($30/mo credit) covers thousands of test queries.
  - Container scales to zero when idle (no charge), spins up on first request.
  - First request after idle ("cold start") takes ~30-60s while the
    container loads PyTorch + the matryoshka model + SPLADE state.
"""
from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# 1. Container image
# -----------------------------------------------------------------------------
# Built reproducibly from a slim Python base, mirrors the Dockerfile.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential", "curl")
    .pip_install(
        # torch >= 2.6 required by transformers' security check (CVE-2025-32434)
        # for loading pytorch_model.bin files like SPLADE's checkpoint.
        "torch==2.6.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        # Pin transformers to 4.x — major-version jumps to 5.x can break model
        # loading. Compatible with sentence-transformers 3.x.
        "transformers>=4.44.0,<5.0.0",
        "sentence-transformers>=3.0.0,<4.0.0",
    )
    .pip_install_from_requirements("requirements-serve.txt")
    .run_commands(
        # Pre-download NLTK data so the first request isn't slow
        "python -c \"import nltk; nltk.download('punkt'); nltk.download('punkt_tab', quiet=True)\""
    )
    # Copy source + configs into the image. Artifacts come from the Volume.
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("configs", remote_path="/app/configs")
)

# -----------------------------------------------------------------------------
# 2. Persistent Volume for artifacts
# -----------------------------------------------------------------------------
# Created once via `modal volume create esci-artifacts`. Populated via
# `modal volume put`. Mounted read-only into the container at /data.
artifacts_volume = modal.Volume.from_name("esci-artifacts", create_if_missing=True)

# -----------------------------------------------------------------------------
# 3. Modal App
# -----------------------------------------------------------------------------
app = modal.App("esci-search")


@app.function(
    image=image,
    volumes={"/data": artifacts_volume},
    cpu=4.0,
    memory=16384,           # 16 GB — pair_df + SPLADE matrix + model are big
    timeout=600,            # generous startup timeout
    min_containers=0,       # scale to zero when idle (set to 1 to keep warm)
    max_containers=2,       # cap blast radius on the free tier
)
@modal.asgi_app()
def fastapi_app():
    """
    The actual ASGI application. Modal wraps this in HTTPS and routes
    requests to a container running our FastAPI app.

    Environment overrides for the app:
      - CE_DEVICE=cpu              (no GPU on this container)
      - ARTIFACTS_DIR=/data/artifacts/systemA
      - PROCESSED_DIR=/data/processed
      - MATRYOSHKA_DIR=/data/artifacts/matryoshka_models
      - MATRYOSHKA_SUBDIR=us
    """
    import os
    import sys

    # Tell main.py where the artifacts are inside the container.
    os.environ.setdefault("CE_DEVICE", "cpu")
    os.environ.setdefault("ARTIFACTS_DIR", "/data/artifacts/systemA")
    os.environ.setdefault("PROCESSED_DIR", "/data/processed")
    os.environ.setdefault("MATRYOSHKA_DIR", "/data/artifacts/matryoshka_models")
    os.environ.setdefault("MATRYOSHKA_SUBDIR", "us")

    # Make sure /app is on the path so `from src.api.main import app` resolves.
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    # Switch CWD to /app so the main.py PROJECT_ROOT calculation works
    # (it walks up from the source file location, not CWD — but the os.chdir
    # in main.py needs a sane working dir).
    try:
        os.chdir("/app")
    except FileNotFoundError:
        pass

    from src.api.main import app as fastapi_application
    return fastapi_application