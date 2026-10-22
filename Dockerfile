FROM rocm/pytorch:latest

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN uv pip install --system --no-deps ultralytics && \
    uv pip install --system pyyaml tqdm pillow matplotlib requests psutil pandas opencv-python-headless scipy && \
    uv pip install --system polars seaborn thop albumentations

WORKDIR /workspace