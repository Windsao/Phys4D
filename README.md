# phys4d User Setup Guide

This document describes how to set up and use phys4d on Linux.

## 1. Prerequisites

Install the following tools before setup:

- `python` (recommended 3.10+, based on project requirements)
- `uv` (Python package/env manager)
- `wget`, `unzip`
- NVIDIA driver + CUDA runtime (required for robot/simulation workflows)

Optional checks:

```bash
python --version
uv --version
wget --version
unzip -v
```

## 2. Install Base Python Environment

Run these commands in the `phys4d/` root directory:

```bash
uv venv .venv
source .venv/bin/activate
uv sync --all-extras
```s

> Keep `.venv` activated for all Python-related commands below.

## 3. Install Robot Dependencies

### 3.1 Install IsaacLab

```bash
cd packages/phys4D/Third_Party/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
./isaaclab.sh -i
```

### 3.2 Install curobo

```bash
cd ../curobo
# If CUDA architecture detection fails, set this first:
export TORCH_CUDA_ARCH_LIST=8.0+PTX
uv pip install -e . --no-build-isolation
uv pip install -U packaging
```

## 4. Install Tactile Modules (Optional)

From repository root with `.venv` activated:

```bash
cd packages/phys4D/Third_Party/TacEx
uv pip install -e source/tacex
uv pip install -e source/tacex_assets
uv pip install -e source/tacex_tasks
```

## 5. Install Isaac Sim (Optional)

If neither of these directories exists on your machine:

- `$HOME/isaacsim`
- `/isaac-sim`

Download and install Isaac Sim (example version `5.1.0`):

```bash
mkdir -p "$HOME/isaacsim"
cd "$HOME/Downloads"
wget https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.1.0-linux-x86_64.zip
unzip isaac-sim-standalone-5.1.0-linux-x86_64.zip -d "$HOME/isaacsim"
cd "$HOME/isaacsim"
./post_install.sh
```

Then create the workspace link:

```bash
cd /path/to/phys4d
ln -s "$HOME/isaacsim" ./_isaac_sim
isaacsim-links --create
```

> If your Isaac Sim path is `/isaac-sim`, use that path as the symlink source instead.

## 6. Quick Setup Order

1. Create and activate `.venv`
2. Run `uv sync --all-extras`
3. Install IsaacLab + curobo
4. (Optional) Install TacEx
5. (Optional) Install Isaac Sim and create `_isaac_sim`
