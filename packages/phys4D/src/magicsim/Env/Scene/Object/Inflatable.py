import torch
import random
import re
import os
import omni.kit.commands
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.api.materials import PreviewSurface

from omni.physx.scripts import particleUtils, physicsUtils
from pxr import UsdPhysics, Sdf, Gf, Usd, UsdGeom, UsdShade, Vt

from magicsim.Env.Utils.path import get_usd_paths_from_folder
from omegaconf import DictConfig
from termcolor import cprint
import numpy as np
import omni.usd


class InflatableObject(SingleXFormPrim):
    """
    InflatableObject class, built using the core particleUtils.
    This approach avoids the deprecated SingleClothPrim and provides a more stable foundation for creating inflatable objects.
    Supports semantic labeling for object identification and scene understanding.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        layout_manager=None,
        primitive_type: str = None,
        layout_info=None,
    ):
        """
        Initializes the InflatableObject, defining its properties and prims on the stage.
        The actual physics application is deferred to the initialize() method.
        """

        super().__init__(prim_path, name=prim_path.split("/")[-1])

        self._stage = get_current_stage()
        self._current_color = None

        if primitive_type == "Plane":
            raise ValueError(
                f"InflatableObject '{prim_path}' does not support 'Plane' primitive type."
                "Inflatable objects require a closed 3D mesh."
            )

        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")
        self.instance_name = self._re_instance_name(self.instance_name)

        self.global_config = config
        self.category_config = config.objects[self.category_name]
        self.instance_config = self.category_config.get(self.instance_name, {})

        category_common_config_val = self.category_config.get("common")
        self.category_common_config = (
            category_common_config_val if category_common_config_val is not None else {}
        )

        if layout_manager and hasattr(layout_manager, "common_config"):
            self.global_common_config = layout_manager.common_config
        else:
            self.global_common_config = (
                self.global_config.objects.common
                if hasattr(self.global_config.objects, "common")
                else {}
            )
        self.visual_cfg = self.instance_config.get("visual", {})
        self.physics_cfg = self.instance_config.get("physics", {})

        self.physics_cfg = self._apply_physics_ratio_randomization(self.physics_cfg)

        self.inst_inflatable_cfg = self.physics_cfg.get("inflatable_config", {})
        self.inst_particle_system_cfg = self.physics_cfg.get("particle_system", {})
        inst_visual_material_cfg = self.visual_cfg.get("visual_material", {})

        self.usd_prim_path = prim_path
        self.usd_path = usd_path
        self.primitive_type = primitive_type
        cprint(
            f"Info: Defining inflatable object at prim path: {self.usd_prim_path}",
            "green",
        )

        self.particle_system_path = Sdf.Path(self.usd_prim_path + "/particleSystem")
        self.particle_material_path = Sdf.Path(self.usd_prim_path + "/particleMaterial")

        self.layout_manager = layout_manager
        self.layout_info = layout_info
        self.env_origin = env_origin.detach().cpu().numpy()

        if self.layout_info:
            self.init_pos = self.layout_info["pos"]
            self.init_ori = self.layout_info["ori"]
            self.init_scale = self.layout_info["scale"]
        else:
            if not self.layout_manager:
                raise RuntimeError(
                    f"LayoutManager is required for {self.usd_prim_path}. All position information must come from LayoutManager."
                )

            env_id = self._extract_env_id_from_prim_path()
            if env_id is None:
                raise ValueError(
                    f"Could not extract env_id from prim path: {self.usd_prim_path}"
                )

            layout_info = self.layout_manager.get_object_layout(
                env_id=env_id, prim_path=self.usd_prim_path
            )
            if layout_info is None:
                raise RuntimeError(
                    f"LayoutManager failed to generate/retrieve layout for {self.usd_prim_path}"
                )

            self.init_pos = layout_info["pos"]
            self.init_ori = layout_info["ori"]
            self.init_scale = layout_info["scale"]

        if usd_path:
            self.mesh_prim_path = find_unique_string_name(
                self.usd_prim_path + "/mesh",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            add_reference_to_stage(usd_path=usd_path, prim_path=self.mesh_prim_path)
            self.geom_prim_path = self._find_first_mesh_in_hierarchy(
                self.mesh_prim_path
            )
            if self.geom_prim_path is None:
                raise RuntimeError(
                    f"Could not find a UsdGeom.Mesh prim under {self.mesh_prim_path}"
                )
        else:
            self.mesh_prim_path = self.usd_prim_path
            self.geom_prim_path = self.usd_prim_path

        mesh_prim = self._stage.GetPrimAtPath(self.geom_prim_path)
        physicsUtils.setup_transform_as_scale_orient_translate(mesh_prim)
        physicsUtils.set_or_add_scale_op(
            mesh_prim,
            Gf.Vec3f([float(v) for v in self.init_scale]),
        )

        visible = self.visual_cfg.get("visible", True)
        if not visible:
            imageable = UsdGeom.Imageable(self.prim)
            imageable.MakeInvisible()

        self.color_list = self.visual_cfg.get("color")
        if self.color_list is not None and isinstance(self.color_list[0], (int, float)):
            self.color_list = [self.color_list]

        self.visual_material_usd_folder = None

        if self.color_list:
            self._current_color = random.choice(self.color_list)
            self._apply_color_material(self._current_color)
        else:
            self.visual_material_usd_folder = inst_visual_material_cfg.get(
                "material_usd_folder", "$MAGICSIM_ASSETS/Material/Garment"
            )
            if self.visual_material_usd_folder is not None:
                self.visual_usd_paths = get_usd_paths_from_folder(
                    folder_path=self.visual_material_usd_folder,
                    skip_keywords=[".thumbs"],
                )
                if self.visual_usd_paths:
                    selected_path = random.choice(self.visual_usd_paths)
                    self._apply_visual_material(selected_path)

        self._handle_semantic_labels()

    def _apply_color_material(self, color):
        """Creates or updates the PreviewSurface material with the specified color."""
        material_path = find_unique_string_name(
            initial_name=f"{self.usd_prim_path}/Looks/color_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        material_prim = get_prim_at_path(material_path)
        if not material_prim:
            material = PreviewSurface(
                prim_path=material_path, color=torch.tensor(color)
            )
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.geom_prim_path,
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )
        else:
            material = PreviewSurface(prim_path=material_path)
            material.set_color(np.array(color))

    def _re_instance_name(self, inst_name):
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        inst_name = f"{cat_name_extracted}_{original_id}"
        return inst_name

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None
        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString
        return None

    def initialize(self):
        """
        Applies physics to the mesh using particleUtils functions.
        """
        particleUtils.add_physx_particle_system(
            stage=self._stage,
            particle_system_path=self.particle_system_path,
            **self.inst_particle_system_cfg,
        )

        particleUtils.add_pbd_particle_material(
            stage=self._stage,
            path=self.particle_material_path,
            **self.physics_cfg.get("particle_material", {}),
        )
        system_prim = self._stage.GetPrimAtPath(self.particle_system_path)
        physicsUtils.add_physics_material_to_prim(
            self._stage, system_prim, self.particle_material_path
        )

        cloth_config = {
            key: value
            for key, value in self.inst_inflatable_cfg.items()
            if key != "particle_mass"
        }
        particleUtils.add_physx_particle_cloth(
            stage=self._stage,
            path=Sdf.Path(self.geom_prim_path),
            dynamic_mesh_path=None,
            particle_system_path=self.particle_system_path,
            **cloth_config,
        )

        mesh_prim_api = UsdGeom.Mesh.Get(self._stage, self.geom_prim_path)
        num_verts = len(mesh_prim_api.GetPointsAttr().Get())
        particle_mass = self.inst_inflatable_cfg.get("particle_mass", 1e-2)
        total_mass = particle_mass * num_verts
        mass_api = UsdPhysics.MassAPI.Apply(mesh_prim_api.GetPrim())
        mass_api.GetMassAttr().Set(total_mass)

        self.set_local_pose(
            translation=self.init_pos,
            orientation=self.init_ori,
        )

    def reset(self, soft=False):
        """Reset inflatable pose using LayoutManager."""
        if not self.layout_manager:
            raise RuntimeError(
                f"LayoutManager is required for {self.usd_prim_path}. All position information must come from LayoutManager."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            print(
                f"Warning: Could not extract env_id for {self.usd_prim_path}. Cannot perform reset."
            )
            return

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self.usd_prim_path, reset_type=reset_type
        )

        if new_layout:
            pos = new_layout["pos"]
            ori = new_layout["ori"]

            pos[2] += random.uniform(-0.0001, 0.0001)
            self.set_local_pose(
                translation=pos,
                orientation=ori,
            )
        else:
            print(
                f"Warning: LayoutManager did not provide new layout for {self.usd_prim_path}. Cannot perform reset."
            )
            return

    def _apply_visual_material(self, material_path: str):
        material_prim_path = find_unique_string_name(
            self.usd_prim_path + "/visual_material", lambda x: not is_prim_path_valid(x)
        )
        add_reference_to_stage(usd_path=material_path, prim_path=material_prim_path)

        visual_material_prim = prims_utils.get_prim_at_path(material_prim_path)
        material_prim = prims_utils.get_prim_children(visual_material_prim)[0]
        material_path_str = material_prim.GetPath().pathString

        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.geom_prim_path,
            material_path=material_path_str,
        )

    def get_current_mesh_points(
        self,
        visualize: bool = False,
        save: bool = False,
        save_path: str = "./pointcloud.ply",
    ):
        """
        Get current mesh vertex positions for this inflatable object.

        Returns (points_world, points_local, pos_world, ori_world).
        """
        mesh_prim = UsdGeom.Mesh.Get(self._stage, self.geom_prim_path)
        if not mesh_prim:
            return np.array([]), np.array([]), None, None

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
                print(f"Error during visualization/saving inflatable point cloud: {e}")

        return points_world, points_local, pos_world, ori_world

    def set_current_mesh_points(
        self, mesh_points: np.ndarray, pos_world=None, ori_world=None
    ):
        """
        Set current mesh vertex positions (local space) back to the mesh. Pose update is optional.
        """
        mesh_prim = UsdGeom.Mesh.Get(self._stage, self.geom_prim_path)
        if not mesh_prim:
            return
        try:
            mesh_prim.GetPointsAttr().Set(
                Vt.Vec3fArray.FromNumpy(np.asarray(mesh_points, dtype=np.float32))
            )
        except Exception as e:
            print(f"Error setting inflatable mesh points: {e}")

    def _extract_env_id_from_prim_path(self):
        """get env_id from prim_path"""
        try:
            parts = self.usd_prim_path.split("/")
            for part in parts:
                if part.startswith("env_"):
                    return int(part.split("_")[1])
        except (ValueError, IndexError):
            pass
        return None

    def _apply_physics_ratio_randomization(self, physics_config):
        """Apply ratio-based randomization to physics parameters.

        Args:
            physics_config: Original physics configuration dictionary

        Returns:
            Modified physics configuration with randomized values
        """

        modified_config = physics_config.copy()

        ratio = modified_config.get("ratio", 1.0)

        if ratio == 1.0:
            return modified_config

        physics_params_to_randomize = [
            "mass",
            "density",
            "linear_velocity",
            "angular_velocity",
        ]

        for param in physics_params_to_randomize:
            if param in modified_config and modified_config[param] is not None:
                original_value = modified_config[param]
                if isinstance(original_value, (int, float)):
                    variation = original_value * (ratio - 1)
                    min_val = original_value - variation
                    max_val = original_value + variation
                    modified_config[param] = random.uniform(min_val, max_val)
                elif isinstance(original_value, list) and len(original_value) > 0:
                    randomized_list = []
                    for val in original_value:
                        if isinstance(val, (int, float)):
                            variation = val * (ratio - 1)
                            min_val = val - variation
                            max_val = val + variation
                            randomized_list.append(random.uniform(min_val, max_val))
                        else:
                            randomized_list.append(val)
                    modified_config[param] = randomized_list

        if "inflatable_config" in modified_config:
            inflatable_config = modified_config["inflatable_config"].copy()
            inflatable_params_to_randomize = [
                "pressure",
                "particle_mass",
                "spring_stretch_stiffness",
                "spring_bend_stiffness",
                "spring_shear_stiffness",
                "spring_damping",
            ]

            for param in inflatable_params_to_randomize:
                if param in inflatable_config and inflatable_config[param] is not None:
                    original_value = inflatable_config[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        inflatable_config[param] = random.uniform(min_val, max_val)

            modified_config["inflatable_config"] = inflatable_config

        if "particle_system" in modified_config:
            particle_system = modified_config["particle_system"].copy()
            particle_system_params_to_randomize = ["contact_offset", "rest_offset"]

            for param in particle_system_params_to_randomize:
                if param in particle_system and particle_system[param] is not None:
                    original_value = particle_system[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        particle_system[param] = random.uniform(min_val, max_val)

            modified_config["particle_system"] = particle_system

        if "particle_material" in modified_config:
            particle_material = modified_config["particle_material"].copy()
            material_params_to_randomize = ["friction"]

            for param in material_params_to_randomize:
                if param in particle_material and particle_material[param] is not None:
                    original_value = particle_material[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        particle_material[param] = random.uniform(min_val, max_val)

            modified_config["particle_material"] = particle_material

        return modified_config

    def destroy(self):
        if is_prim_path_valid(self.usd_prim_path):
            imageable = UsdGeom.Imageable.Get(self._stage, self.usd_prim_path)
            if imageable:
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

        if self.primitive_type:
            return self.primitive_type

        if not self.usd_path:
            return ""

        regex_pattern = self.category_config.get("semantic_regex_pattern", r".*")
        regex_replacement = self.category_config.get("semantic_regex_repl", r"\g<0>")
        filename = os.path.basename(self.usd_path)
        filename_without_ext = os.path.splitext(filename)[0]
        return re.sub(regex_pattern, regex_replacement, filename_without_ext)

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Get the state of the inflatable object.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.

        Returns:
            Dictionary containing:
                - root_pose: torch.Tensor, shape (7,), position (3) and quaternion (4)
                - asset_info: dict with usd_path and primitive_type
        """
        try:
            import omni.usd

            world_tf = omni.usd.get_world_transform_matrix(self.prim)
            translation = np.array(world_tf.ExtractTranslation(), dtype=np.float32)
            rot_quat = world_tf.ExtractRotationQuat()
            orientation = np.array(
                [rot_quat.GetReal(), *rot_quat.GetImaginary()], dtype=np.float32
            )
        except (AttributeError, RuntimeError) as e:
            try:
                translation = self.get_translation()
                orientation = self.get_orientation()
                translation = np.array(translation, dtype=np.float32)
                orientation = np.array(orientation, dtype=np.float32)
            except (AttributeError, RuntimeError):
                print(
                    f"Warning: Failed to get pose for {self.prim_path if hasattr(self, 'prim_path') else 'unknown'}: {e}"
                )
                translation = np.zeros(3, dtype=np.float32)
                orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=torch.float32)
        if not isinstance(orientation, torch.Tensor):
            orientation = torch.tensor(orientation, dtype=torch.float32)

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

        asset_info = {
            "usd_path": self.usd_path if hasattr(self, "usd_path") else None,
            "primitive_type": self.primitive_type
            if hasattr(self, "primitive_type")
            else None,
        }

        return {
            "root_pose": root_pose,
            "asset_info": asset_info,
        }
