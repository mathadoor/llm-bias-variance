# LLM Bias-Variance Trade-off

Small-scale study: treat the instruction writer as the "learner". Training set -> instructions -> model.
An LLM writes instructions from different input-output samples (proxy for fresh-minded experts); we measure the
bias and variance of the resulting zero-shot predictions.

## Layout
```
shared/          reusable code (data loading, prompting, metrics, MLflow helpers)
configs/         experiment configs (YAML)
notebooks/       experiments; refactor into shared/ as they stabilise
requirements.txt
.Dockerfile
docker-compose.yml
```

## Setup and Usage
```
docker compose up --build   # JupyterLab at http://localhost:8888
```
The project folder is bind-mounted at `/workspace`, so notebooks and changes to `shared/` and `configs/` persist on your host.
Datasets are cached in your host's `~/.cache/huggingface` (also bind-mounted), so they are shared with your other repos and tools. Set `HF_CACHE_DIR` in `.env` to use another location. Create the folder first (`mkdir -p ~/.cache/huggingface`) so Docker doesn't create it root-owned.
On Linux, if your user id isn't 1000, set `HOST_UID` and `HOST_GID` in `.env` (`id -u`, `id -g`) before building.

**GPU:** the container uses the host's NVIDIA GPU(s). This needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`). Verify with `docker compose run --rm lab python -c "import torch; print(torch.cuda.is_available())"`.

**MLflow:** by default the container logs to a tracking server running on your host at port 5000
(`http://host.docker.internal:5000`). The server must listen on `0.0.0.0`, e.g. `mlflow server --host 0.0.0.0 --port 5000`.
Override via `.env`:
```
MLFLOW_TRACKING_URI=http://my-server:5000
# or, with no server, log to a local folder:
MLFLOW_TRACKING_URI=file:///workspace/mlruns
```

