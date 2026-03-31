from tacex import GelSightSensorCfg
from tacex.simulation_approaches.fem_based import ManiSkillSimulatorCfg
from tacex.simulation_approaches.gpu_taxim import TaximSimulatorCfg

from tacex_assets import TACEX_ASSETS_DATA_DIR

from .gsmini_cfg import GelSightMiniCfg

"""Configuration for simulating the Gelsight Mini via Taxim and ManiSkill-Approach."""


GELSIGHT_MINI_TAXIM_FEM_CFG = GelSightMiniCfg()
GELSIGHT_MINI_TAXIM_FEM_CFG = GELSIGHT_MINI_TAXIM_FEM_CFG.replace(
    sensor_camera_cfg=GelSightSensorCfg.SensorCameraCfg(
        prim_path_appendix="/Camera",
        update_period=0,
        resolution=(32, 24),
        data_types=["depth"],
        clipping_range=(0.024, 0.029),
    ),
    update_period=0.01,
    data_types=["tactile_rgb", "marker_motion"],
    optical_sim_cfg=TaximSimulatorCfg(
        calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/GelSight_Mini/calibs/640x480",
        gelpad_height=GELSIGHT_MINI_TAXIM_FEM_CFG.gelpad_dimensions.height,
        gelpad_to_camera_min_distance=0.024,
        with_shadow=False,
        tactile_img_res=(320, 240),
        device="cuda",
    ),
    marker_motion_sim_cfg=ManiSkillSimulatorCfg(),
    compute_indentation_depth_class="optical_sim",
    device="cuda",
)


















































