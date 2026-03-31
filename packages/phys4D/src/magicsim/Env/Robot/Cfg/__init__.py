from magicsim.Env.Robot.Cfg.Base import RobotCfg
from magicsim.Env.Robot.Cfg.Manipulator.Franka import FrankaCfg
from magicsim.Env.Robot.Cfg.Manipulator.UR10 import UR10Cfg

from magicsim.Env.Robot.Cfg.Manipulator.FrankaTactile import FrankaTactileCfg
from magicsim.Env.Robot.Cfg.Mobile.novaCarter import NovaCarterCfg
from magicsim.Env.Robot.Cfg.Mobile.leatherback import LeatherbackCfg
from magicsim.Env.Robot.Cfg.Manipulator.FrankaUMI import FrankaUMICfg

ROBOT_DICT: dict[str, type[RobotCfg]] = {
    "franka": FrankaCfg,
    "ur10": UR10Cfg,
    "franka_tactile": FrankaTactileCfg,
    "novacarter": NovaCarterCfg,
    "leatherback": LeatherbackCfg,
    "franka_umi": FrankaUMICfg,
}
