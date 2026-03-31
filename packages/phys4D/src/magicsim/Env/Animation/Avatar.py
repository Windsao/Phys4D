from omegaconf import DictConfig
import torch
from isaacsim.core.utils.stage import get_current_stage
from pxr import Usd, Gf
from isaacsim.core.prims import SingleGeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage
import numpy as np
import omni.anim.graph.core as ag
import carb
from typing import List, Tuple, Optional
from omni.anim.people.scripts.navigation_manager import NavigationManager
from omni.anim.people.scripts.global_queue_manager import GlobalQueueManager


class SimpleSit:
    """Simplified Sit command - sit in place using set_variable only"""

    def __init__(
        self,
        character,
        command,
        character_name,
        navigation_manager,
        command_id,
        update_metadata_callback_fn,
    ):
        self.character = character
        self.command = command
        self.character_name = character_name
        self.duration = float(command[1]) if len(command) > 1 else 5.0
        self.sit_time = 0
        self.is_setup = False
        self.finished = False

    def get_command_name(self):
        return "Sit"

    def setup(self):
        self.character.set_variable("Action", "Sit")
        self.is_setup = True
        carb.log_info(f"{self.character_name} sitting in place for {self.duration}s")

    def execute(self, dt):
        if self.finished:
            return True
        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def update(self, dt):
        self.sit_time += dt
        if self.sit_time >= self.duration:
            self.character.set_variable("Action", "None")
            self.finished = True
            return True
        return False

    def force_quit_command(self):
        self.character.set_variable("Action", "None")


class SimpleTalk:
    """Simplified Talk command - talk in place using set_variable only"""

    def __init__(
        self,
        character,
        command,
        character_name,
        navigation_manager,
        command_id,
        update_metadata_callback_fn,
    ):
        self.character = character
        self.command = command
        self.character_name = character_name
        self.duration = float(command[1]) if len(command) > 1 else 5.0
        self.talk_time = 0
        self.is_setup = False
        self.finished = False

    def get_command_name(self):
        return "Talk"

    def setup(self):
        self.character.set_variable("Action", "Talk")
        self.is_setup = True
        carb.log_info(f"{self.character_name} talking in place for {self.duration}s")

    def execute(self, dt):
        if self.finished:
            return True
        if not self.is_setup:
            self.setup()
        return self.update(dt)

    def update(self, dt):
        self.talk_time += dt
        if self.talk_time >= self.duration:
            self.character.set_variable("Action", "None")
            self.finished = True
            return True
        return False

    def force_quit_command(self):
        self.character.set_variable("Action", "None")


class Avatar(SingleGeometryPrim):
    """Avatar class to manage individual avatar instances in the simulation."""

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        layout_manager=None,
        layout_info=None,
        collision: bool = True,
    ):
        self.usd_path = usd_path
        self.config = config
        self.env_origin = env_origin
        self.stage = get_current_stage()
        self.layout_manager = layout_manager
        self.layout_info = layout_info

        prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Failed to load USD from {usd_path} to {prim_path}")

        if layout_info:
            pos_from_layout = layout_info["pos"]

            init_pos = (
                np.array(pos_from_layout, dtype=float)
                + np.array(env_origin, dtype=float)
            ).tolist()
            init_ori = layout_info.get("ori", [1.0, 0.0, 0.0, 0.0])
            init_scale = layout_info.get("scale", [1.0, 1.0, 1.0])
        else:
            init_pos = np.array(env_origin, dtype=float).tolist()
            init_ori = [1.0, 0.0, 0.0, 0.0]
            init_scale = [1.0, 1.0, 1.0]

        super().__init__(
            prim_path=prim_path,
            name=prim_path.split("/")[-1],
            translation=init_pos,
            orientation=init_ori,
            scale=init_scale,
            visible=True,
            collision=collision,
        )

        self.skelroot_prim = self.get_skel_root_prim()

        self.character = None
        self.character_name = prim_path.split("/")[-1]
        self.navigation_manager: Optional[NavigationManager] = None
        self.queue_manager: Optional[GlobalQueueManager] = None
        self.current_command = None
        self.commands: List[Tuple] = []
        self.is_initialized = False

    def get_skel_root_prim(self):
        for prim in Usd.PrimRange(self.prim):
            if prim.GetTypeName() == "SkelRoot":
                return prim
        else:
            raise ValueError(f"SkelRoot prim not found under {self.prim_path}")

    def set_world_pose(self, position, orientation):
        """Set world pose using GeometryPrim's standard method.

        Args:
            position: Position as [x, y, z] list or array
            orientation: Orientation as quaternion [w, x, y, z] or Euler angles [rx, ry, rz]
        """

        if len(orientation) == 3:
            rx, ry, rz = (float(v) for v in orientation)
            quat = (
                Gf.Rotation(Gf.Vec3d(1, 0, 0), rx)
                * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
                * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz)
            )
            quat = quat.GetQuat()

            orientation_quat = np.array([quat.GetReal(), *quat.GetImaginary()])
        else:
            orientation_quat = np.array(orientation, dtype=float)

        position_array = np.array(position, dtype=float)

        super().set_world_pose(position=position_array, orientation=orientation_quat)

    def reset(self, soft: bool = False):
        """Reset avatar pose using LayoutManager."""
        if not self.layout_manager:
            raise RuntimeError(
                f"No layout_manager for avatar {self.prim_path}. Cannot reset."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            raise ValueError(
                f"Could not extract env_id for {self.prim_path}. Cannot reset."
            )

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self.prim_path, reset_type=reset_type
        )

        if not new_layout:
            raise RuntimeError(
                f"LayoutManager did not provide new layout for {self.prim_path}."
            )

        pos = new_layout["pos"]
        ori = new_layout["ori"]

        world_pos = (
            np.array(pos, dtype=float) + np.array(self.env_origin, dtype=float)
        ).tolist()
        self.set_world_pose(position=world_pos, orientation=ori)

    def _extract_env_id_from_prim_path(self):
        """Extract env_id from prim_path."""
        try:
            parts = self.prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def init_character(self):
        """Initialize animation graph character object (can only be called at runtime)"""
        if self.is_initialized:
            return True

        self.character = ag.get_character(str(self.skelroot_prim.GetPrimPath()))
        if self.character is None:
            carb.log_warn(
                f"Cannot get character object: {self.skelroot_prim.GetPrimPath()}"
            )
            return False

        self.navigation_manager = NavigationManager(
            str(self.skelroot_prim.GetPrimPath()),
            navmesh_enabled=False,
            dynamic_avoidance_enabled=False,
        )

        self.queue_manager = GlobalQueueManager.get_instance()

        self.character.set_variable("Action", "None")

        self.is_initialized = True
        carb.log_info(f"Character initialized: {self.character_name}")
        return True

    def inject_command(self, command_list: List, execute_immediately: bool = True):
        """
        Inject commands to the character

        Args:
            command_list: Command list, format: [["CommandType", "param1", ...], ...]
            execute_immediately: Whether to execute immediately
        """
        cmd_array = []
        for command in command_list:
            if isinstance(command, list):
                cmd_array.append((None, command))
            elif isinstance(command, str):
                words = command.strip().split()
                cmd_array.append((None, words))

        if execute_immediately:
            if self.commands and cmd_array:
                self.commands[1:1] = cmd_array
            else:
                self.commands[0:0] = cmd_array
        else:
            self.commands.extend(cmd_array)

        carb.log_info(
            f"Commands injected to {self.character_name}: {len(cmd_array)} commands"
        )

    def execute_command(self, delta_time: float):
        """Execute command queue"""
        while not self.current_command:
            if not self.commands:
                return

            next_cmd_pair = self.commands[0]
            command_id, command = next_cmd_pair

            if len(command) < 1:
                self.commands.pop(0)
                continue

            self.current_command = self._create_command_object(command_id, command)

            if self.current_command:
                carb.log_info(f"Start executing command: {command}")
            else:
                self.commands.pop(0)

        if self.current_command:
            try:
                if self.current_command.execute(delta_time):
                    carb.log_info(
                        f"Command completed: {self.current_command.get_command_name()}"
                    )
                    self.commands.pop(0)
                    self.current_command = None
            except Exception as e:
                carb.log_error(f"Command execution error: {e}")
                self.commands.pop(0)
                self.current_command = None

    def _create_command_object(self, command_id, command):
        """Create command object"""
        from omni.anim.people.scripts.commands.goto import GoTo
        from omni.anim.people.scripts.commands.idle import Idle
        from omni.anim.people.scripts.commands.look_around import LookAround

        command_params = {
            "character": self.character,
            "command": command,
            "character_name": str(self.character_name),
            "navigation_manager": self.navigation_manager,
            "command_id": command_id,
            "update_metadata_callback_fn": self._dummy_metadata_callback,
        }

        if len(command) < 1:
            return None

        command_type = command[0]

        if command_type == "GoTo":
            return GoTo(**command_params)
        elif command_type == "Idle":
            return Idle(**command_params)
        elif command_type == "LookAround":
            return LookAround(**command_params)
        elif command_type == "Sit":
            return SimpleSit(**command_params)
        elif command_type == "Talk":
            return SimpleTalk(**command_params)
        else:
            carb.log_warn(f"Unknown command type: {command_type}")
            return None

    def _dummy_metadata_callback(self, agent_name, data_name, data_value):
        """Metadata callback (placeholder)"""
        pass

    def end_current_command(self):
        """Force end current command"""
        if self.current_command is not None:
            self.current_command.force_quit_command()
            self.current_command = None
            carb.log_info(f"Current command ended for {self.character_name}")

    def replace_command(self, command_list: List):
        """Replace all commands with new command list"""

        self.commands.clear()

        self.end_current_command()

        self.inject_command(command_list, execute_immediately=True)
        carb.log_info(f"Commands replaced for {self.character_name}")

    def clear_commands(self):
        """Clear all commands in queue"""
        self.commands.clear()
        carb.log_info(f"Commands cleared for {self.character_name}")

    def get_current_action(self) -> str:
        """Get current executing action name"""
        if self.current_command:
            return self.current_command.get_command_name()
        return "None"

    def get_command_queue_length(self) -> int:
        """Get number of commands in queue"""
        return len(self.commands)

    def on_update(self, current_time: float, delta_time: float):
        """Update every frame"""
        if self.character is None:
            if not self.init_character():
                return

        if self.navigation_manager:
            self.navigation_manager.publish_character_positions(delta_time, 0.5)

        if self.commands:
            self.execute_command(delta_time)
