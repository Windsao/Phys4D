TacEx currently consists of four extensions:
- tacex
- tacex_assets
- tacex_tasks
- tacex_uipc


This is the heart of TacEx - the "Tactile" in TacEx.
This extension enables the simulation of GelSight sensors inside Isaac Sim and
implements approaches for simulation Vision-Based-Tactile-Sensors.

Currently we have:
- Tactile RGB sim with GPU accelerated Taxim (https:
- FOTS marker sim




This extension is responsible for the assets we use in TacEx.
Here we place USD files and configfiles.

The extension enables easy access to the files.
For example, you can use the following snippet to refer to the assets:

```python
from tacex_assets import TACEX_ASSETS_DATA_DIR

ball = f"{TACEX_ASSETS_DATA_DIR}/Props/ball_wood.usd"
```


Contains the files for our RL tasks.

Currently we have the environments:


Integration of libuipc for Isaac Lab.
