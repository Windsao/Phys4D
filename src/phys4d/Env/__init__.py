import gymnasium as gym


from phys4d.Env.BallAndBlockEnv import BallAndBlockEnv
from phys4d.Env.BallCollideEnv import BallCollideEnv
from phys4d.Env.BallHitsDuckEnv import BallHitsDuckEnv
from phys4d.Env.BallHitsNothingEnv import BallHitsNothingEnv
from phys4d.Env.BallInBasketEnv import BallInBasketEnv
from phys4d.Env.BallRampEnv import BallRampEnv
from phys4d.Env.BallRollsOffEnv import BallRollsOffEnv
from phys4d.Env.BallRollsOnGlassEnv import BallRollsOnGlassEnv
from phys4d.Env.BallTrainEnv import BallTrainEnv
from phys4d.Env.BlockDominoEnv import BlockDominoEnv
from phys4d.Env.DominoInJuiceEnv import DominoInJuiceEnv
from phys4d.Env.DominoWithSpaceEnv import DominoWithSpaceEnv
from phys4d.Env.DuckAndDominoEnv import DuckAndDominoEnv
from phys4d.Env.DuckFallsInBoxEnv import DuckFallsInBoxEnv
from phys4d.Env.DuckStaticEnv import DuckStaticEnv
from phys4d.Env.LightOnBlockEnv import LightOnBlockEnv
from phys4d.Env.LightOnMugBlockEnv import LightOnMugBlockEnv
from phys4d.Env.LightOnMugEnv import LightOnMugEnv
from phys4d.Env.LightOnSculptureEnv import LightOnSculptureEnv
from phys4d.Env.MarbleRunXEnv import MarbleRunXEnv
from phys4d.Env.MarbleRunYEnv import MarbleRunYEnv
from phys4d.Env.MirrorBallFallEnv import MirrorBallFallEnv
from phys4d.Env.MirrorBallRotateEnv import MirrorBallRotateEnv
from phys4d.Env.MirrorTeapotRotateEnv import MirrorTeapotRotateEnv
from phys4d.Env.PotatoInWaterEnv import PotatoInWaterEnv
from phys4d.Env.RollBehindBoxEnv import RollBehindBoxEnv
from phys4d.Env.RollFrontBoxEnv import RollFrontBoxEnv
from phys4d.Env.RollInBoxEnv import RollInBoxEnv
from phys4d.Env.RollingReflectionEnv import RollingReflectionEnv
from phys4d.Env.SilkCoverEnv import SilkCoverEnv
from phys4d.Env.SmileyBallRotatesEnv import SmileyBallRotatesEnv
from phys4d.Env.StableBlocksEnv import StableBlocksEnv
from phys4d.Env.TeapotRotatesEnv import TeapotRotatesEnv
from phys4d.Env.TwoBallsPassEnv import TwoBallsPassEnv
from phys4d.Env.UnstableBlockStackEnv import UnstableBlockStackEnv
from phys4d.Env.WeightOnCeramicEnv import WeightOnCeramicEnv
from phys4d.Env.WeightOnPillowEnv import WeightOnPillowEnv
from phys4d.Env.WeightProtectsDuckEnv import WeightProtectsDuckEnv
from phys4d.Env.RotorGsEnv import RotorGsEnv
from phys4d.Env.RotorGsMoveCameraEnv import RotorGsMoveCameraEnv


gym.register(
    id="BallAndBlockEnv-V0",
    entry_point="phys4d.Env.BallAndBlockEnv:BallAndBlockEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BallCollideEnv-V0",
    entry_point="phys4d.Env.BallCollideEnv:BallCollideEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BallHitsDuckEnv-V0",
    entry_point="phys4d.Env.BallHitsDuckEnv:BallHitsDuckEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BallHitsNothingEnv-V0",
    entry_point="phys4d.Env.BallHitsNothingEnv:BallHitsNothingEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BallInBasketEnv-V0",
    entry_point="phys4d.Env.BallInBasketEnv:BallInBasketEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BallRampEnv-V0",
    entry_point="phys4d.Env.BallRampEnv:BallRampEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BallRollsOffEnv-V0",
    entry_point="phys4d.Env.BallRollsOffEnv:BallRollsOffEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BallRollsOnGlassEnv-V0",
    entry_point="phys4d.Env.BallRollsOnGlassEnv:BallRollsOnGlassEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BallTrainEnv-V0",
    entry_point="phys4d.Env.BallTrainEnv:BallTrainEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="BlockDominoEnv-V0",
    entry_point="phys4d.Env.BlockDominoEnv:BlockDominoEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="DominoInJuiceEnv-V0",
    entry_point="phys4d.Env.DominoInJuiceEnv:DominoInJuiceEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="DuckFallsInBoxEnv-V0",
    entry_point="phys4d.Env.DuckFallsInBoxEnv:DuckFallsInBoxEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="DuckStaticEnv-V0",
    entry_point="phys4d.Env.DuckStaticEnv:DuckStaticEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LightOnBlockEnv-V0",
    entry_point="phys4d.Env.LightOnBlockEnv:LightOnBlockEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LightOnMugBlockEnv-V0",
    entry_point="phys4d.Env.LightOnMugBlockEnv:LightOnMugBlockEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LightOnMugEnv-V0",
    entry_point="phys4d.Env.LightOnMugEnv:LightOnMugEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="LightOnSculptureEnv-V0",
    entry_point="phys4d.Env.LightOnSculptureEnv:LightOnSculptureEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MarbleRunXEnv-V0",
    entry_point="phys4d.Env.MarbleRunXEnv:MarbleRunXEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MarbleRunYEnv-V0",
    entry_point="phys4d.Env.MarbleRunYEnv:MarbleRunYEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MirrorBallFallEnv-V0",
    entry_point="phys4d.Env.MirrorBallFallEnv:MirrorBallFallEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MirrorBallRotateEnv-V0",
    entry_point="phys4d.Env.MirrorBallRotateEnv:MirrorBallRotateEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="PotatoInWaterEnv-V0",
    entry_point="phys4d.Env.PotatoInWaterEnv:PotatoInWaterEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="MirrorTeapotRotateEnv-V0",
    entry_point="phys4d.Env.MirrorTeapotRotateEnv:MirrorTeapotRotateEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="RollBehindBoxEnv-V0",
    entry_point="phys4d.Env.RollBehindBoxEnv:RollBehindBoxEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="RollFrontBoxEnv-V0",
    entry_point="phys4d.Env.RollFrontBoxEnv:RollFrontBoxEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="RollInBoxEnv-V0",
    entry_point="phys4d.Env.RollInBoxEnv:RollInBoxEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="RollingReflectionEnv-V0",
    entry_point="phys4d.Env.RollingReflectionEnv:RollingReflectionEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="SilkCoverEnv-V0",
    entry_point="phys4d.Env.SilkCoverEnv:SilkCoverEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="SmileyBallRotatesEnv-V0",
    entry_point="phys4d.Env.SmileyBallRotatesEnv:SmileyBallRotatesEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="StableBlocksEnv-V0",
    entry_point="phys4d.Env.StableBlocksEnv:StableBlocksEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="TeapotRotatesEnv-V0",
    entry_point="phys4d.Env.TeapotRotatesEnv:TeapotRotatesEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="TwoBallsPassEnv-V0",
    entry_point="phys4d.Env.TwoBallsPassEnv:TwoBallsPassEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="UnstableBlockStackEnv-V0",
    entry_point="phys4d.Env.UnstableBlockStackEnv:UnstableBlockStackEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="WeightOnCeramicEnv-V0",
    entry_point="phys4d.Env.WeightOnCeramicEnv:WeightOnCeramicEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="WeightOnPillowEnv-V0",
    entry_point="phys4d.Env.WeightOnPillowEnv:WeightOnPillowEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="WeightProtectsDuckEnv-V0",
    entry_point="phys4d.Env.WeightProtectsDuckEnv:WeightProtectsDuckEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="DominoWithSpaceEnv-V0",
    entry_point="phys4d.Env.DominoWithSpaceEnv:DominoWithSpaceEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="DuckAndDominoEnv-V0",
    entry_point="phys4d.Env.DuckAndDominoEnv:DuckAndDominoEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="RotorGsEnv-V0",
    entry_point="phys4d.Env.RotorGsEnv:RotorGsEnv",
    disable_env_checker=True,
    order_enforce=False,
)

gym.register(
    id="RotorGsMoveCameraEnv-V0",
    entry_point="phys4d.Env.RotorGsMoveCameraEnv:RotorGsMoveCameraEnv",
    disable_env_checker=True,
    order_enforce=False,
)

__all__ = [
    "BallAndBlockEnv",
    "BallCollideEnv",
    "BallHitsDuckEnv",
    "BallHitsNothingEnv",
    "BallInBasketEnv",
    "BallRampEnv",
    "BallRollsOffEnv",
    "BallRollsOnGlassEnv",
    "BallTrainEnv",
    "BlockDominoEnv",
    "DominoInJuiceEnv",
    "DominoWithSpaceEnv",
    "DuckAndDominoEnv",
    "DuckFallsInBoxEnv",
    "DuckStaticEnv",
    "LightOnBlockEnv",
    "LightOnMugBlockEnv",
    "LightOnMugEnv",
    "LightOnSculptureEnv",
    "MarbleRunXEnv",
    "MarbleRunYEnv",
    "MirrorBallFallEnv",
    "MirrorBallRotateEnv",
    "MirrorTeapotRotateEnv",
    "PotatoInWaterEnv",
    "RollBehindBoxEnv",
    "RollFrontBoxEnv",
    "RollInBoxEnv",
    "RollingReflectionEnv",
    "SilkCoverEnv",
    "SmileyBallRotatesEnv",
    "StableBlocksEnv",
    "TeapotRotatesEnv",
    "TwoBallsPassEnv",
    "UnstableBlockStackEnv",
    "WeightOnCeramicEnv",
    "WeightOnPillowEnv",
    "WeightProtectsDuckEnv",
    "RotorGsEnv",
    "RotorGsMoveCameraEnv",
]
