FROM python:3.12-slim

# Match the host user so files written to the mounted project are not root-owned
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} dev && useradd -m -u ${UID} -g ${GID} dev

# C compiler: Triton (torch.compile, bitsandbytes, etc.) JIT-compiles kernel launchers
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Mount point for the host's Hugging Face cache (see docker-compose.yml)
ENV HF_HOME=/home/dev/.cache/huggingface
RUN mkdir -p ${HF_HOME} && chown -R dev:dev /home/dev/.cache

USER dev
# Project code is bind-mounted here by docker-compose.yml, not copied into the image
WORKDIR /workspace
ENV PYTHONPATH=/workspace

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token="]
