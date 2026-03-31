

[![IsaacSim](https:
[![Isaac Lab](https:
[![Python](https:
[![Linux platform](https:
<!-- [![Windows platform](https:
[![pre-commit](https:
[![License](https:

**Keywords:** tactile sensing, gelsight, isaaclab, vision-based-tactile-sensor, vbts, reinforcement learning

> [!note]
> **Preview Release**:
>
> The framework is under active development and currently in its beta phase.
> If you encounter bugs or have suggestions on how the framework can be improved, please tell us about them (e.g. via [Issues](https:



**TacEx** brings **Vision-Based Tactile Sensor (VBTS)** into Isaac Sim/Lab.

The framework integrates multiple simulation approaches for VBTS's and aims to be modular and extendable.
Components can be easily switched out, added and modified.

Currently, only the **GelSight Mini** is supported, but you can also easily add your own sensor (guide coming soon). We also plan to add more VBTS types later.


- [GPU accelerated Tactile RGB simulation](https:
- Marker Motion Simulation via [FOTS](https:
- Integration of [UIPC](https:
- Marker Motion Simulation with FEM soft body based on the simulator used by the [ManiSkill-ViTac challenge](https:


Checkout the [website](https:



> [!NOTE]
> TacEx currently works with **Isaac Sim 4.5** and **IsaacLab 2.1.1**.
> The installation was tested on Ubuntu 22.04 with a 4090 GPU and Driver Version 550.163.01 + Cuda 12.4.

**0.** Make sure that you have **git-lfs**:

```bash

git lfs install
```

**1.** Clone this repository and its submodules:
```bash
git clone --recurse-submodules https:
cd TacEx
```

Then **install TacEx** [locally](docs/source/installation/Local-Installation.md)
or build a [Docker Container](docs/source/installation/Docker-Container-Setup.md).



Contributions of any kind are, of course, very welcome.
Be it suggestions, feedback, bug reports or pull requests.

Let's work together to advance tactile sensing in robotics!!!


```bibtex
@article{nguyen2024tacexgelsighttactilesimulation,
      title={TacEx: GelSight Tactile Simulation in Isaac Sim -- Combining Soft-Body and Visuotactile Simulators},
      author={Duc Huy Nguyen and Tim Schneider and Guillaume Duret and Alap Kshirsagar and Boris Belousov and Jan Peters},
      year={2024},
      eprint={2411.04776},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https:
}
```



TacEx is built upon code from
- [Isaac Lab](https:
- [Taxim](https:
- [FOTS](https:
- [UIPC](https:
- [ManiSkill-ViTac challenge](https:
