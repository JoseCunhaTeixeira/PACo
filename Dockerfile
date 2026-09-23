# PACo's image: the MCP server (the default command), the agent's chat (paco-agent) and the
# evaluation (paco-evaluate). Profiles are read from /data/input and results written to
# /data/output: mount a folder at /data.

FROM python:3.14-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /bin/uv
# sigpipe is installed from GitHub, at the commit uv.lock pins; bayesbay (sigpipe's MCMC) has no
# wheel for Python 3.14 and compiles C++ extensions. Neither tool goes into the final image.
RUN apt-get update && apt-get install --yes --no-install-recommends git g++ \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first: they change less often than PACo's code, so their layer is reused.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.14-slim
RUN useradd --uid 1000 --create-home paco
COPY --from=build /app/.venv /app/.venv
ENV PATH=/app/.venv/bin:$PATH \
    PACO_INPUT_DIR=/data/input \
    PACO_OUTPUT_DIR=/data/output \
    PACO_LOG_DIR=/data/output/agent_logs \
    PACO_EVALUATION_DIR=/data/output/evaluations \
    PACO_HOST=0.0.0.0 \
    MPLBACKEND=Agg
USER paco
WORKDIR /home/paco
EXPOSE 8000
CMD ["paco-server"]
