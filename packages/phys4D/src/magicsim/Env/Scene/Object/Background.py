from omegaconf import DictConfig
import carb
from pxr import UsdLux, Sdf
from isaacsim.core.utils.stage import get_current_stage
import random
from magicsim.Env.Utils.path import resolve_path


class Background:
    """
    A class to create Background in magicsim. This is used for setting up hdr environment lighting and background textures.
    """

    def __init__(self, prim_path: str, config: DictConfig):
        """
        Constructor: Initializes an instance of the Background class.

        Args:
            prim_path (str): The path for the background prim on the USD stage.
            config (DictConfig): The configuration from the YAML file, containing parameters for the background.
        """
        self.prim_path = prim_path
        self.config = config
        self.light_prim = None

        self.create()
        self.initialize()

    def create(self):
        """
        Create a DomeLight in the USD stage.
        This function defines a new light prim via the API, not from a pre-existing USD file.
        """
        render_mode = carb.settings.get_settings().get("/rtx/rendermode")
        rt_subframes = carb.settings.get_settings().get("/omni/replicator/RTSubframes")
        if render_mode == "RaytracedLighting" and (
            rt_subframes is None or rt_subframes < 3
        ):
            carb.log_warn(
                "`/omni/replicator/RTSubframes` must be > 3 to avoid blank textures while randomizing dome "
                f"light texture. RTSubframes has been automatically increased from {rt_subframes} to 3"
            )
            carb.settings.get_settings().set("/omni/replicator/RTSubframes", 3)

        stage = get_current_stage()

        self.light_prim: UsdLux.DomeLight = UsdLux.DomeLight.Define(
            stage, self.prim_path
        )
        return self.light_prim

    def initialize(self):
        """Initialize the light with the configuration after the first hard reset."""

        self.reset()

    def reset(self):
        """
        Soft Reset Light and Perform Domain Randomization Here.
        This method randomizes the light's intensity and background texture.
        """
        if not self.light_prim:
            print("Error: Light prim has not been created yet.")
            return

        intensity_range = self.config.intensity

        random_intensity = random.uniform(intensity_range[0], intensity_range[1])

        self.light_prim.GetIntensityAttr().Set(random_intensity)

        texture_list = self.config.texture

        texture_path = resolve_path(random.choice(texture_list))
        self.light_prim.GetTextureFileAttr().Set(Sdf.AssetPath(texture_path))
