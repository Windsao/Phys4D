import torch
import random
import re
import os
import numpy as np
import omni.kit.commands
from pxr import UsdGeom, Sdf, UsdShade, Usd, Vt, Gf
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.api.materials import PreviewSurface
from omegaconf import DictConfig
from isaacsim.core.simulation_manager import SimulationManager
from magicsim.Env.Utils.path import get_usd_paths_from_folder
import isaacsim.core.utils.prims as prims_utils
import omni.usd


class ArticulationObject(SingleArticulation):
    """
    ArticulationObject class that wraps the Isaac Sim SingleArticulation functionality.
    This class inherits from the Isaac Sim SingleArticulation class and can be extended.
    Supports semantic labeling for object identification and scene understanding.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        layout_manager=None,
        layout_info=None,
    ):
        """
        Initialize the ArticulationObject with position, orientation, and configuration.
        """
        if usd_path:
            prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        else:
            prim = get_prim_at_path(prim_path)

        if not prim or not prim.IsValid():
            error_message = (
                f"Failed to load USD from {usd_path} to {prim_path}"
                if usd_path
                else f"Failed to find an existing prim at path {prim_path}"
            )
            raise RuntimeError(error_message)

        self.stage = get_current_stage()
        self._current_color = None
        self._prim_path = prim_path
        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")
        self.instance_name = self._re_instance_name(self.instance_name)

        self.global_config = config

        if layout_manager and hasattr(layout_manager, "common_config"):
            self.global_common_config = layout_manager.common_config
        else:
            self.global_common_config = (
                self.global_config.objects.common
                if hasattr(self.global_config.objects, "common")
                else {}
            )

        self.category_config = config.objects[self.category_name]
        category_common_config_val = self.category_config.get("common")
        self.category_common_config = (
            category_common_config_val if category_common_config_val is not None else {}
        )

        self.instance_config = self.category_config.get(self.instance_name, {})
        self.visual_cfg = self.instance_config.get("visual", {})
        self.physics_cfg = self.instance_config.get("physics", {})

        inst_visual_material_cfg = self.visual_cfg.get("visual_material", {})

        self.usd_prim_path = prim_path
        self.usd_path = usd_path
        self.prim_name = prim_path_parts[-1]
        self.layout_manager = layout_manager
        self.layout_info = layout_info

        if self.layout_info:
            pos_from_layout = self.layout_info["pos"]

            if torch.is_tensor(pos_from_layout):
                pos_from_layout = pos_from_layout.cpu().numpy()
            self.init_pos = pos_from_layout
            ori_from_layout = self.layout_info["ori"]
            if torch.is_tensor(ori_from_layout):
                ori_from_layout = ori_from_layout.cpu().numpy()
            self.init_ori = ori_from_layout
            scale_from_layout = self.layout_info["scale"]
            if torch.is_tensor(scale_from_layout):
                scale_from_layout = scale_from_layout.cpu().numpy()
            self.init_scale = scale_from_layout
        else:
            if not self.layout_manager:
                raise RuntimeError(
                    f"LayoutManager is required for {self._prim_path}. All position information must come from LayoutManager."
                )

            env_id = self._extract_env_id_from_prim_path()
            if env_id is None:
                raise ValueError(
                    f"Could not extract env_id from prim path: {self._prim_path}"
                )

            layout_info = self.layout_manager.get_object_layout(
                env_id=env_id, prim_path=self._prim_path
            )
            if layout_info is None:
                layout_info = self.layout_manager.register_object_and_get_layout(
                    env_id=env_id,
                    prim_path=self._prim_path,
                    cat_name=self.category_name,
                    inst_cfg=self.instance_config,
                    cat_spec=self.category_config,
                    asset_to_spawn=None,
                )

            if layout_info is None:
                raise RuntimeError(
                    f"LayoutManager failed to generate/retrieve layout for {self._prim_path}"
                )

            pos_from_layout = layout_info["pos"]

            if torch.is_tensor(pos_from_layout):
                pos_from_layout = pos_from_layout.cpu().numpy()
            self.init_pos = pos_from_layout
            ori_from_layout = layout_info["ori"]
            if torch.is_tensor(ori_from_layout):
                ori_from_layout = ori_from_layout.cpu().numpy()
            self.init_ori = ori_from_layout
            scale_from_layout = layout_info["scale"]
            if torch.is_tensor(scale_from_layout):
                scale_from_layout = scale_from_layout.cpu().numpy()
            self.init_scale = scale_from_layout

        super().__init__(
            prim_path=self._prim_path,
            name=self.prim_name,
            translation=self.init_pos,
            orientation=self.init_ori,
            scale=self.init_scale,
        )

        self.color_list = self.visual_cfg.get("color")
        if self.color_list is not None and isinstance(self.color_list[0], (int, float)):
            self.color_list = [self.color_list]

        self.visual_material_usd_folder = None
        self.color_material_path = None

        if self.color_list:
            self._current_color = random.choice(self.color_list)
            self._apply_color_material(self._current_color)
        else:
            color = self.visual_cfg.get("color")
            if color:
                material_path = find_unique_string_name(
                    initial_name=f"{self.usd_prim_path}/Looks/color_material",
                    is_unique_fn=lambda x: not is_prim_path_valid(x),
                )
                self.color_material_path = material_path
                material = PreviewSurface(
                    prim_path=material_path, color=torch.tensor(color)
                )

                for child_prim in Usd.PrimRange(self.prim):
                    if (
                        child_prim.IsA(UsdGeom.Mesh)
                        or child_prim.IsA(UsdGeom.Capsule)
                        or child_prim.IsA(UsdGeom.Sphere)
                        or child_prim.IsA(UsdGeom.Cube)
                        or child_prim.IsA(UsdGeom.Cylinder)
                        or child_prim.IsA(UsdGeom.Cone)
                    ):
                        omni.kit.commands.execute(
                            "BindMaterialCommand",
                            prim_path=child_prim.GetPath(),
                            material_path=material_path,
                            strength=UsdShade.Tokens.strongerThanDescendants,
                        )
            else:
                self.visual_material_usd_folder = inst_visual_material_cfg.get(
                    "material_usd_folder"
                )
                if self.visual_material_usd_folder is not None:
                    self.visual_usd_paths = get_usd_paths_from_folder(
                        folder_path=self.visual_material_usd_folder,
                        skip_keywords=[".thumbs"],
                    )
                    if self.visual_usd_paths:
                        selected_path = random.choice(self.visual_usd_paths)

                        self._apply_visual_material_from_file_original_logic(
                            selected_path
                        )
                    else:
                        print(
                            f"⚠️ Warning: No visual materials found in '{self.visual_material_usd_folder}'. Skipping material application for {self.usd_prim_path}."
                        )
        visible = self.visual_cfg.get("visible", True)
        if not visible:
            imageable = UsdGeom.Imageable(self.prim)
            if imageable:
                imageable.MakeInvisible()

        self._handle_semantic_labels()

        try:
            if self.layout_manager:
                env_id_for_assign = self._extract_env_id_from_prim_path()
                if env_id_for_assign is not None:
                    self.layout_manager._initialize_category_list(
                        env_id_for_assign, self.category_name
                    )
                    self.layout_manager._assign_object_to_category(
                        self.category_name, env_id_for_assign, self
                    )
        except Exception:
            pass

    def _apply_color_material(self, color):
        """Creates or updates the PreviewSurface material with the specified color."""
        if self.color_material_path is None:
            self.color_material_path = find_unique_string_name(
                initial_name=f"{self.usd_prim_path}/Looks/color_material",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )

        material_prim = get_prim_at_path(self.color_material_path)
        if not material_prim:
            material = PreviewSurface(
                prim_path=self.color_material_path, color=torch.tensor(color)
            )
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.prim.GetPath(),
                material_path=self.color_material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )
        else:
            material = PreviewSurface(prim_path=self.color_material_path)
            try:
                material.set_color(np.array(color))
            except Exception as e:
                print(
                    f"Error setting color for material {self.color_material_path}: {e}"
                )

    def _apply_visual_material_from_file_original_logic(self, material_path: str):
        """Applies a visual material from a USD file using the original logic."""
        visual_material_path = find_unique_string_name(
            self.usd_prim_path + "/visual_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        add_reference_to_stage(usd_path=material_path, prim_path=visual_material_path)
        visual_material_ref_prim = prims_utils.get_prim_at_path(visual_material_path)
        if not visual_material_ref_prim or not visual_material_ref_prim.IsValid():
            return
        material_children = prims_utils.get_prim_children(visual_material_ref_prim)
        if not material_children:
            return

        material_prim = material_children[0]
        material_prim_path_str = material_prim.GetPath().pathString

        for child_prim in Usd.PrimRange(self.prim):
            if (
                child_prim.IsA(UsdGeom.Mesh)
                or child_prim.IsA(UsdGeom.Capsule)
                or child_prim.IsA(UsdGeom.Sphere)
                or child_prim.IsA(UsdGeom.Cube)
                or child_prim.IsA(UsdGeom.Cylinder)
                or child_prim.IsA(UsdGeom.Cone)
            ):
                omni.kit.commands.execute(
                    "BindMaterialCommand",
                    prim_path=child_prim.GetPath(),
                    material_path=material_prim_path_str,
                    strength=UsdShade.Tokens.strongerThanDescendants,
                )

    def _re_instance_name(self, inst_name):
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        inst_name = f"{cat_name_extracted}_{original_id}"
        return inst_name

    def hide_prim(self, prim_path: str):
        try:
            path = Sdf.Path(prim_path)
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                return
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                imageable.MakeInvisible()
            else:
                visibility_attribute = prim.GetAttribute("visibility")
                if visibility_attribute:
                    visibility_attribute.Set("invisible")
        except Exception as e:
            print(f"Warning: Failed to hide prim {prim_path}: {e}")

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None
        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString
        return None

    def get_current_mesh_points(
        self,
        visualize: bool = False,
        save: bool = False,
        save_path: str = "./pointcloud.ply",
    ):
        """
        Get current mesh vertex positions for the first mesh under this articulation.

        Returns (points_world, points_local, pos_world, ori_world).
        """
        mesh_path = self._find_first_mesh_in_hierarchy(self._prim_path)
        if mesh_path is None:
            return np.array([]), np.array([]), None, None

        mesh_prim = UsdGeom.Mesh.Get(self.stage, mesh_path)
        points_local = np.array(mesh_prim.GetPointsAttr().Get(), dtype=np.float32)
        world_tf = omni.usd.get_world_transform_matrix(mesh_prim.GetPrim())

        points_world = np.array(
            [
                list(
                    world_tf.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
                )
                for p in points_local
            ],
            dtype=np.float32,
        )

        rot_quat = world_tf.ExtractRotationQuat()
        pos_world = np.array(world_tf.ExtractTranslation(), dtype=np.float32)
        ori_world = np.array(
            [rot_quat.GetReal(), *rot_quat.GetImaginary()], dtype=np.float32
        )

        if visualize or save:
            try:
                import open3d as o3d

                if points_world.size > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_world)
                    if visualize:
                        o3d.visualization.draw_geometries([pcd])
                    if save:
                        o3d.io.write_point_cloud(save_path, pcd)
            except Exception as e:
                print(
                    f"Error during visualization/saving articulation point cloud: {e}"
                )

        return points_world, points_local, pos_world, ori_world

    def set_current_mesh_points(
        self, mesh_points: np.ndarray, pos_world=None, ori_world=None
    ):
        """
        Set current mesh vertex positions (local space) back to the first mesh under this articulation.
        """
        mesh_path = self._find_first_mesh_in_hierarchy(self._prim_path)
        if mesh_path is None:
            return
        mesh_prim = UsdGeom.Mesh.Get(self.stage, mesh_path)
        try:
            mesh_prim.GetPointsAttr().Set(
                Vt.Vec3fArray.FromNumpy(np.asarray(mesh_points, dtype=np.float32))
            )
        except Exception as e:
            print(f"Error setting articulation mesh points: {e}")

    def initialize(self):
        self.physics_sim_view = SimulationManager.get_physics_sim_view()
        super().initialize(physics_sim_view=self.physics_sim_view)
        self.upper_joint_positions = self.dof_properties["upper"].copy()
        self.lower_joint_positions = self.dof_properties["lower"].copy()
        self.initial_joint_positions = self.get_current_joint_positions()

    def get_current_joint_positions(self):
        return self.get_joint_positions()

    def set_current_joint_positions(self, positions):
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32)
        self.set_joint_positions(positions)

    def _extract_env_id_from_prim_path(self):
        """ "get env_id from prim_path"""
        try:
            parts = self._prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def reset(self, soft=False):
        """Reset articulation pose using LayoutManager."""
        self.set_current_joint_positions(self.initial_joint_positions)

        if not self.layout_manager:
            raise RuntimeError(
                f"LayoutManager is required for {self._prim_path}. All position information must come from LayoutManager."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            print(
                f"Warning: Could not extract env_id for {self._prim_path}. Cannot perform reset."
            )
            return

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self._prim_path, reset_type=reset_type
        )

        if not new_layout:
            print(
                f"Warning: LayoutManager did not provide new layout for {self._prim_path}. Cannot perform reset."
            )
            return

        pos = new_layout["pos"]

        if torch.is_tensor(pos):
            pos = pos.cpu().numpy()
        ori = new_layout["ori"]
        if torch.is_tensor(ori):
            ori = ori.cpu().numpy()
        scale = new_layout["scale"]
        if torch.is_tensor(scale):
            scale = scale.cpu().numpy()

        self.set_local_pose(pos, ori)
        self.set_local_scale(np.array(scale))

        visible = self.visual_cfg.get("visible", True)
        imageable = UsdGeom.Imageable(self.prim)
        if imageable:
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

    def _handle_semantic_labels(self):
        """Manage semantic labeling: clear existing labels and apply new ones."""

        remove_labels(self.prim, include_descendants=True)
        semantic_label = self._get_semantic_label()
        if semantic_label:
            add_labels(self.prim, [semantic_label])
            self.semantic_label = semantic_label

    def _get_semantic_label(self) -> str:
        """Generate semantic label from configuration or USD filename."""

        if (
            hasattr(self.category_config, "semantic_label")
            and self.category_config.semantic_label
        ):
            return self.category_config.semantic_label

        if not self.usd_path:
            return ""

        regex_pattern = self.category_config.get("semantic_regex_pattern", r".*")
        regex_replacement = self.category_config.get("semantic_regex_repl", r"\g<0>")
        filename = os.path.basename(self.usd_path)
        filename_without_ext = os.path.splitext(filename)[0]
        return re.sub(regex_pattern, regex_replacement, filename_without_ext)

    def reset_hard(self, soft: bool = False):
        """Reset articulation pose and optionally randomize appearance.

        Mirrors Rigid.reset_hard: fetch new layout from LayoutManager, apply
        translation/orientation/scale, then randomize color or visual material.
        """

        self.set_current_joint_positions(self.initial_joint_positions)

        if not self.layout_manager:
            raise RuntimeError(
                f"LayoutManager is required for {self._prim_path}. All position information must come from LayoutManager."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            print(
                f"Warning: Could not extract env_id for {self._prim_path}. Cannot perform reset."
            )
            return

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self._prim_path, reset_type=reset_type
        )

        if not new_layout:
            print(
                f"Warning: LayoutManager did not provide new layout for {self._prim_path}. Cannot perform reset."
            )
            return

        pos = new_layout["pos"]

        if torch.is_tensor(pos):
            pos = pos.cpu().numpy()
        ori = new_layout["ori"]
        if torch.is_tensor(ori):
            ori = ori.cpu().numpy()
        scale = new_layout["scale"]
        if torch.is_tensor(scale):
            scale = scale.cpu().numpy()

        self.set_local_pose(pos, ori)
        self.set_local_scale(np.array(scale))

        visible = self.visual_cfg.get("visible", True)
        imageable = UsdGeom.Imageable(self.prim)
        if imageable:
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()

        if self.color_list:
            random_color = random.choice(self.color_list)
            self._apply_color_material(random_color)
        elif (
            self.color_list is None
            and hasattr(self, "visual_material_usd_folder")
            and self.visual_material_usd_folder
            and hasattr(self, "visual_usd_paths")
            and self.visual_usd_paths
        ):
            selected_path = random.choice(self.visual_usd_paths)
            self._apply_visual_material_from_file_original_logic(selected_path)

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Get the state of the articulation object.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.

        Returns:
            Dictionary containing:
                - root_pose: torch.Tensor, shape (7,), position (3) and quaternion (4)
                - root_velocity: torch.Tensor, shape (6,), linear velocity (3) and angular velocity (3)
                - joint_position: torch.Tensor, shape (num_joints,), joint positions
                - joint_velocity: torch.Tensor, shape (num_joints,), joint velocities
                - asset_info: dict with usd_path and primitive_type
        """
        try:
            world_tf = omni.usd.get_world_transform_matrix(self.prim)
            translation = np.array(world_tf.ExtractTranslation(), dtype=np.float32)
            rot_quat = world_tf.ExtractRotationQuat()
            orientation = np.array(
                [rot_quat.GetReal(), *rot_quat.GetImaginary()], dtype=np.float32
            )
        except (AttributeError, RuntimeError) as e:
            try:
                translation, orientation = self.get_current_pose()
                translation = np.array(translation, dtype=np.float32)
                orientation = np.array(orientation, dtype=np.float32)
            except (AttributeError, RuntimeError):
                print(f"Warning: Failed to get pose for {self._prim_path}: {e}")
                translation = np.zeros(3, dtype=np.float32)
                orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(
                translation,
                dtype=torch.float32,
                device=self.device if hasattr(self, "device") else "cpu",
            )
        if not isinstance(orientation, torch.Tensor):
            orientation = torch.tensor(
                orientation, dtype=torch.float32, device=translation.device
            )

        if translation.dim() > 1:
            translation = translation.squeeze()
        if orientation.dim() > 1:
            orientation = orientation.squeeze()

        root_pose = torch.cat([translation, orientation])

        if is_relative and hasattr(self, "env_origin"):
            env_origin_tensor = (
                torch.tensor(
                    self.env_origin, dtype=torch.float32, device=root_pose.device
                )
                if isinstance(self.env_origin, np.ndarray)
                else self.env_origin
            )
            if env_origin_tensor.dim() == 0:
                env_origin_tensor = env_origin_tensor.unsqueeze(0)
            if env_origin_tensor.shape[0] < 3:
                env_origin_tensor = torch.cat(
                    [
                        env_origin_tensor,
                        torch.zeros(
                            3 - env_origin_tensor.shape[0],
                            device=env_origin_tensor.device,
                        ),
                    ]
                )
            root_pose[:3] -= env_origin_tensor[:3]

        try:
            if (
                hasattr(self, "_articulation_prim_view")
                and self._articulation_prim_view is not None
            ):
                linear_vel = self._articulation_prim_view.get_linear_velocities()
                angular_vel = self._articulation_prim_view.get_angular_velocities()
                if linear_vel is not None and angular_vel is not None:
                    if not isinstance(linear_vel, torch.Tensor):
                        linear_vel = torch.tensor(
                            linear_vel, dtype=torch.float32, device=root_pose.device
                        )
                    if not isinstance(angular_vel, torch.Tensor):
                        angular_vel = torch.tensor(
                            angular_vel, dtype=torch.float32, device=root_pose.device
                        )
                    if linear_vel.dim() > 1:
                        linear_vel = linear_vel.squeeze()
                    if angular_vel.dim() > 1:
                        angular_vel = angular_vel.squeeze()
                    root_velocity = torch.cat([linear_vel, angular_vel])
                else:
                    root_velocity = torch.zeros(
                        6, dtype=torch.float32, device=root_pose.device
                    )
            else:
                linear_vel = self.get_linear_velocity()
                angular_vel = self.get_angular_velocity()
                if not isinstance(linear_vel, torch.Tensor):
                    linear_vel = torch.tensor(
                        linear_vel, dtype=torch.float32, device=root_pose.device
                    )
                if not isinstance(angular_vel, torch.Tensor):
                    angular_vel = torch.tensor(
                        angular_vel, dtype=torch.float32, device=root_pose.device
                    )
                if linear_vel.dim() > 1:
                    linear_vel = linear_vel.squeeze()
                if angular_vel.dim() > 1:
                    angular_vel = angular_vel.squeeze()
                root_velocity = torch.cat([linear_vel, angular_vel])
        except (AttributeError, RuntimeError):
            root_velocity = torch.zeros(6, dtype=torch.float32, device=root_pose.device)

        joint_position = self.get_current_joint_positions()
        if not isinstance(joint_position, torch.Tensor):
            joint_position = torch.tensor(
                joint_position, dtype=torch.float32, device=root_pose.device
            )
        if joint_position.dim() > 1:
            joint_position = joint_position.squeeze()

        try:
            joint_velocity = self.get_joint_velocities()
            if not isinstance(joint_velocity, torch.Tensor):
                joint_velocity = torch.tensor(
                    joint_velocity, dtype=torch.float32, device=root_pose.device
                )
            if joint_velocity.dim() > 1:
                joint_velocity = joint_velocity.squeeze()
        except (AttributeError, RuntimeError):
            joint_velocity = torch.zeros_like(joint_position)

        asset_info = {
            "usd_path": self.usd_path if hasattr(self, "usd_path") else None,
            "primitive_type": None,
        }

        return {
            "root_pose": root_pose,
            "root_velocity": root_velocity,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "asset_info": asset_info,
        }
