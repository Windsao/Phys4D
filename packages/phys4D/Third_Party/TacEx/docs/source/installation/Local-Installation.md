If you have a working Isaac Lab environment, you can directly [install TacEx](Local-Installation
Otherwise, **you need to install Isaac Sim 4.5 and Isaac Lab 2.1.1**.
Below is a quick summary, but here is the [full installation guide](https:

<details>
<summary>Quick summary for Installing Isaac Sim and Isaac Lab for Ubuntu 22.04</summary>

> [!note]
> To install Isaac Sim for Ubuntu 20.04 follow the [binary installation guide](https:



```bash

conda create -n env_isaaclab python=3.10
conda activate env_isaaclab

pip install torch==2.5.1 torchvision==0.20.1 --index-url https:
pip install --upgrade pip

pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https:
```

> verify that the Isaac Sim installation works by calling `isaacsim` in the terminal



```bash

sudo apt install cmake build-essential
git clone https:
cd IsaacLab

git checkout v2.1.1

conda activate env_isaaclab

./isaaclab.sh --install
```

To verify the Isaac Lab Installation:

```bash
conda activate env_isaaclab
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

</details>



**0.** If you haven't already done so, clone the repository and its submodules:

```bash

git lfs install
git clone --recurse-submodules https:
cd TacEx
```
**1.** Activate the Isaac Env
```bash
conda activate env_isaaclab
```

**2.** Install the core packages of TacEx
```bash

./tacex.sh -i
```

> You can install the extensions one by one via e.g. `python -m pip install -e source/tacex_uipc`

**3.** Verify that TacEx works by running an example:

```bash
python ./scripts/demos/tactile_sim_approaches/check_taxim_sim.py --debug_vis
```

And here is an RL example:
```bash
python ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-v0 --num_envs 512 --enable_cameras
```
> You can view the sensor output in the IsaacLab Tab: `Scene Debug Visualization > Observations > sensor_output`


The `tacex_uipc` package is responsible for the [UIPC](https:

**1.** Install the [libuipc dependencies](https:
* If not installed yet, install Vcpkg

```bash
mkdir ~/Toolchain
cd ~/Toolchain
git clone https:
cd vcpkg
./bootstrap-vcpkg.sh -disableMetrics
```

* Set the System Environment Variable  `CMAKE_TOOLCHAIN_FILE` to let CMake detect Vcpkg. If you installed it like above, you can do this:

```bash

export CMAKE_TOOLCHAIN_FILE="$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

* We also need `CMake 3.26`, `GCC 11.4` and `Cuda 12.4` to build libuipc. Install this into the Isaac Sim python env:

```bash

conda activate env_isaaclab
conda env update -n env_isaaclab --file ./source/tacex_uipc/libuipc/conda/env.yaml
```
> If Cuda 12.4 does not work for, try updating your Nvidia drivers or try to use an older Cuda version by adjusting the env.yaml file (e.g. Cuda 12.2).

**2.** Install `tacex_uipc`
```bash

conda activate env_isaaclab
pip install -e source/tacex_uipc -v
```
> You can also install all TacEx packages with `./tacex.sh -i all`.

**3.** Verify that the `tacex_uipc` works by running an example:

```bash
python ./scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py --num_envs 1 --debug_vis --env uipc
```



There is a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```



Full setup Guide can be found [here](https:

In a nutshell:

1. Run VSCode Tasks, by pressing `Ctrl+Shift+P`and selecting `Tasks: Run Task`
2. Select `setup_python_env` in the drop down menu.

Now you should have

- `.vscode/launch.json`, which contains the launch configurations for debugging python code.
- `.vscode/settings.json`, which contains the settings for the python interpreter and the python environment.
