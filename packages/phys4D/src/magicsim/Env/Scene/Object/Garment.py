import torch
import numpy as np
import random
import re
import os
import omni.kit.commands
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
from isaacsim.core.utils.prims import get_prim_at_path, is_prim_path_valid
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.api.materials.preview_surface import PreviewSurface
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Vt, Usd, UsdGeom, UsdShade
from magicsim.Env.Utils.path import get_usd_paths_from_folder
from omegaconf import DictConfig


class GarmentObject(SingleClothPrim):
    """
    GarmentObject class that wraps the Isaac Sim SingleCloth prim functionality.
    This class inherits from the Isaac Sim SingleClothPrim class and can be extended
    to add custom garment-specific behaviors.
    Supports semantic labeling for object identification and scene understanding.
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        config: DictConfig,
        env_origin: torch.Tensor,
        primitive_type: str = None,
        layout_manager=None,
        layout_info=None,
    ):
        """
        Initialize the GarmentObject with position, orientation, and configuration.

        Args:
            prim_path: Path to the prim in the stage
            usd_path: Path to the USD asset file for this object
            config: Configuration dictionary containing object properties
            env_origin: Origin position of the environment
            primitive_type: Type of primitive (only 'Plane' is supported)
        """
        if primitive_type is not None and primitive_type != "Plane":
            raise ValueError(
                f"GarmentObject '{prim_path}' only supports the 'Plane' primitive type, "
                f"but received '{primitive_type}'"
            )

        prim_path_parts = prim_path.split("/")
        self.category_name = prim_path_parts[-2]
        self.instance_name = prim_path_parts[-1]
        self.env_name = prim_path_parts[-4]
        self.num_per_env = config.objects[self.category_name].get("num_per_env")
        self.instance_name = self._re_instance_name(self.instance_name)

        self.primitive_type = primitive_type
        self.global_config = config
        self.category_config = config.objects[self.category_name]
        self.instance_config = self.category_config.get(self.instance_name, {})
        self.stage = get_current_stage()
        self._current_color = None

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

        self.inst_garment_cfg = self.physics_cfg.get("garment_config", {})
        self.inst_particle_material_cfg = self.physics_cfg.get("particle_material", {})
        self.inst_particle_system_cfg = self.physics_cfg.get("particle_system", {})
        self.inst_visual_material_cfg = self.visual_cfg.get("visual_material", {})

        self.usd_prim_path = prim_path
        self.usd_path = usd_path
        self.prim_name = prim_path.split("/")[-1]
        self.config = config
        self.objects_config = config.get("objects")
        self.layout_manager = layout_manager

        self.env_origin = env_origin.detach().cpu().numpy()
        self.layout_manager = layout_manager
        self.layout_info = layout_info

        if self.layout_info:
            pos_from_layout = self.layout_info["pos"]
            self.init_pos = (
                np.array(pos_from_layout, dtype=np.float32) + self.env_origin
            )
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

            pos_from_layout = layout_info["pos"]
            self.init_pos = (
                np.array(pos_from_layout, dtype=np.float32) + self.env_origin
            )
            self.init_ori = layout_info["ori"]
            self.init_scale = layout_info["scale"]

        if usd_path:
            add_reference_to_stage(usd_path=usd_path, prim_path=self.usd_prim_path)
            self.mesh_prim_path = self._find_first_mesh_in_hierarchy(self.usd_prim_path)
            if self.mesh_prim_path is None:
                raise RuntimeError(
                    f"Could not find a UsdGeom.Mesh prim under the referenced asset at {self.usd_prim_path}"
                )
        else:
            self.mesh_prim_path = self.usd_prim_path

        interaction_flag = self.category_config.get("interaction_with_fluid", False)
        if interaction_flag:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/particle_system"
            )
        else:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/garment_particle_system"
            )

        if is_prim_path_valid(self.particle_system_path):
            self.particle_system = SingleParticleSystem(
                prim_path=self.particle_system_path
            )
        else:
            self.particle_system = SingleParticleSystem(
                prim_path=self.particle_system_path,
                particle_system_enabled=self.inst_particle_system_cfg.get(
                    "particle_system_enabled", True
                ),
                enable_ccd=self.inst_particle_system_cfg.get("enable_ccd", True),
                solver_position_iteration_count=self.inst_particle_system_cfg.get(
                    "solver_position_iteration_count", 16
                ),
                max_depenetration_velocity=self.inst_particle_system_cfg.get(
                    "max_depenetration_velocity", None
                ),
                global_self_collision_enabled=self.inst_particle_system_cfg.get(
                    "global_self_collision_enabled", True
                ),
                non_particle_collision_enabled=self.inst_particle_system_cfg.get(
                    "non_particle_collision_enabled", True
                ),
                contact_offset=self.inst_particle_system_cfg.get(
                    "contact_offset", 0.01
                ),
                rest_offset=self.inst_particle_system_cfg.get("rest_offset", 0.0075),
                particle_contact_offset=self.inst_particle_system_cfg.get(
                    "particle_contact_offset", 0.01
                ),
                fluid_rest_offset=self.inst_particle_system_cfg.get(
                    "fluid_rest_offset", 0.0075
                ),
                solid_rest_offset=self.inst_particle_system_cfg.get(
                    "solid_rest_offset", 0.0075
                ),
                wind=self.inst_particle_system_cfg.get("wind", None),
                max_neighborhood=self.inst_particle_system_cfg.get(
                    "max_neighborhood", None
                ),
                max_velocity=self.inst_particle_system_cfg.get("max_velocity", None),
            )

        self.particle_material_path = find_unique_string_name(
            self.usd_prim_path + "/particle_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        self.particle_material = ParticleMaterial(
            prim_path=self.particle_material_path,
            adhesion=self.inst_particle_material_cfg.get("adhesion", 0.1),
            adhesion_offset_scale=self.inst_particle_material_cfg.get(
                "adhesion_offset_scale", 0.0
            ),
            cohesion=self.inst_particle_material_cfg.get("cohesion", 0.0),
            particle_adhesion_scale=self.inst_particle_material_cfg.get(
                "particle_adhesion_scale", 0.5
            ),
            particle_friction_scale=self.inst_particle_material_cfg.get(
                "particle_friction_scale", 0.5
            ),
            drag=self.inst_particle_material_cfg.get("drag", 0.0),
            lift=self.inst_particle_material_cfg.get("lift", 0.0),
            friction=self.inst_particle_material_cfg.get("friction", 10.0),
            damping=self.inst_particle_material_cfg.get("damping", 0.0),
            gravity_scale=self.inst_particle_material_cfg.get("gravity_scale", 1.0),
            viscosity=self.inst_particle_material_cfg.get("viscosity", None),
            vorticity_confinement=self.inst_particle_material_cfg.get(
                "vorticity_confinement", None
            ),
            surface_tension=self.inst_particle_material_cfg.get(
                "surface_tension", None
            ),
        )

        super().__init__(
            name=self.usd_prim_path,
            scale=self.init_scale,
            prim_path=self.mesh_prim_path,
            particle_system=self.particle_system,
            particle_material=self.particle_material,
            particle_mass=self.inst_garment_cfg.get("particle_mass", 1e-2),
            self_collision=self.inst_garment_cfg.get("self_collision", True),
            self_collision_filter=self.inst_garment_cfg.get(
                "self_collision_filter", True
            ),
            stretch_stiffness=self.inst_garment_cfg.get("stretch_stiffness", 1e8),
            bend_stiffness=self.inst_garment_cfg.get("bend_stiffness", 1000.0),
            shear_stiffness=self.inst_garment_cfg.get("shear_stiffness", 1000.0),
            spring_damping=self.inst_garment_cfg.get("spring_damping", 10.0),
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
            self.visual_material_usd_folder = self.inst_visual_material_cfg.get(
                "material_usd_folder", "$MAGICSIM_ASSETS/Material/Garment"
            )
            if self.visual_material_usd_folder is not None:
                self.visual_usd_paths = get_usd_paths_from_folder(
                    folder_path=self.visual_material_usd_folder,
                    skip_keywords=[".thumbs"],
                )
                if self.visual_usd_paths:
                    selected_indices = torch.randint(
                        low=0,
                        high=len(self.visual_usd_paths),
                        size=(1,),
                    ).tolist()
                    self.visual_usd_path = self.visual_usd_paths[selected_indices[0]]
                    self._apply_visual_material(self.visual_usd_path)

        self.set_world_pose(position=self.init_pos, orientation=self.init_ori)

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
                prim_path=self.mesh_prim_path,
                material_path=material_path,
                strength=UsdShade.Tokens.strongerThanDescendants,
            )

            mesh_prim_to_bind = prims_utils.get_prim_at_path(self.mesh_prim_path)
            if mesh_prim_to_bind:
                garment_submesh = prims_utils.get_prim_children(mesh_prim_to_bind)
                if len(garment_submesh) > 0:
                    for sub_prim in garment_submesh:
                        if sub_prim.IsA(UsdGeom.Gprim):
                            omni.kit.commands.execute(
                                "BindMaterialCommand",
                                prim_path=sub_prim.GetPath(),
                                material_path=material_path,
                                strength=UsdShade.Tokens.strongerThanDescendants,
                            )
        else:
            material = PreviewSurface(prim_path=material_path)
            material.set_color(np.array(color))

    def _find_first_mesh_in_hierarchy(self, prim_path: str) -> str:
        """Recursively searches for the first prim of type UsdGeom.Mesh under the given path."""
        start_prim = get_prim_at_path(prim_path)
        if not start_prim:
            return None

        for prim in Usd.PrimRange(start_prim):
            if prim.IsA(UsdGeom.Mesh):
                return prim.GetPath().pathString

        return None

    def _re_instance_name(self, inst_name):
        """Reformats the instance name to ensure consistent numbering."""
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        return f"{cat_name_extracted}_{original_id}"

    def initialize(self):
        """
        Initialize the object by capturing initial particle information
        and setting up initial state.
        """
        self._get_initial_info()

    def reset(self, soft=False):
        """
        Perform reset by restoring initial particle positions and setting new pose using LayoutManager.

        Args:
            soft: If True, use soft reset ranges; otherwise use initial ranges
        """

        if self._device == "cpu":
            self._prim.GetAttribute("points").Set(
                Vt.Vec3fArray.FromNumpy(self.initial_points_positions)
            )
        else:
            if hasattr(self, "_cloth_prim_view") and self._cloth_prim_view:
                if isinstance(self.initial_points_positions, np.ndarray):
                    initial_pos_tensor = torch.from_numpy(
                        self.initial_points_positions
                    ).to(self._device)
                    if initial_pos_tensor.ndim == 2:
                        initial_pos_tensor = initial_pos_tensor.unsqueeze(0)
                else:
                    initial_pos_tensor = self.initial_points_positions.to(self._device)

                expected_shape_prefix = (
                    self._cloth_prim_view.get_world_positions().shape[:-1]
                )
                if initial_pos_tensor.shape[:-1] != expected_shape_prefix:
                    if len(expected_shape_prefix) == 2 and initial_pos_tensor.ndim == 2:
                        initial_pos_tensor = initial_pos_tensor.unsqueeze(0)
                try:
                    self._cloth_prim_view.set_world_positions(initial_pos_tensor)
                except Exception as e:
                    print(f"Error setting world positions in reset: {e}")
                    print(f"  Expected shape prefix: {expected_shape_prefix}")
                    print(f"  Provided tensor shape: {initial_pos_tensor.shape}")
            else:
                print(
                    f"Warning: _cloth_prim_view not initialized for {self.name} on device {self._device}. Skipping particle position reset."
                )
        if self.layout_manager is not None:
            env_id = self._extract_env_id_from_prim_path()
            if env_id is not None:
                reset_type = "soft" if soft else "hard"
                new_layout = self.layout_manager.generate_new_layout(
                    env_id=env_id, prim_path=self.usd_prim_path, reset_type=reset_type
                )
                if new_layout:
                    position = new_layout["pos"]
                    orientation = new_layout["ori"]
                    scale = new_layout["scale"]

                    position = np.array(position, dtype=np.float32) + self.env_origin
                    position[2] += random.uniform(-0.0001, 0.0001)
                    self.set_world_pose(position, orientation)
                    if hasattr(self, "set_local_scale"):
                        self.set_local_scale(scale)
                    return

    def get_current_mesh_points(
        self, visualize=False, save=False, save_path="./pointcloud.ply"
    ):
        """
        Get the current mesh points of the garment.

        Args:
            visualize: Whether to visualize the mesh points using Open3D
            save: Whether to save the mesh points to a file
            save_path: Path to save the point cloud if save=True

        Returns:
            transformed_points: Mesh points in world space
            mesh_points: Original mesh points in local space
            pos_world: World position (CPU only)
            ori_world: World orientation (CPU only)
        """
        if self._device == "cpu":
            pos_world, ori_world = self.get_world_pose()
            scale_world = self.get_world_scale()
            mesh_points = self._get_points_pose().detach().cpu().numpy()
            transformed_mesh_points = self.transform_points(
                mesh_points,
                pos_world.detach().cpu().numpy(),
                ori_world.detach().cpu().numpy(),
                scale_world.detach().cpu().numpy(),
            )
        else:
            mesh_points = (
                self._cloth_prim_view.get_world_positions()
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
            )
            transformed_mesh_points = mesh_points
            pos_world = None
            ori_world = None

        if visualize or save:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(transformed_mesh_points)

            if visualize:
                o3d.visualization.draw_geometries([pcd])
            if save:
                o3d.io.write_point_cloud(save_path, pcd)

        return transformed_mesh_points, mesh_points, pos_world, ori_world

    def set_current_mesh_points(self, mesh_points, pos_world, ori_world):
        """
        Set the current mesh points of the garment object.

        Args:
            mesh_points: Original mesh points in local space
            pos_world: World position (required for CPU device)
            ori_world: World orientation (required for CPU device)
        """
        if self._device == "cpu":
            if pos_world is None or ori_world is None:
                raise ValueError(
                    "pos_world and ori_world must be provided for CPU device"
                )
            self._prim.GetAttribute("points").Set(Vt.Vec3fArray.FromNumpy(mesh_points))
            self.set_world_pose(pos_world, ori_world)
        else:
            current_mesh_points = (
                torch.from_numpy(mesh_points).to(self._device).unsqueeze(0)
            )
            self._cloth_prim_view.set_world_positions(current_mesh_points)

    def _apply_visual_material(self, material_path: str):
        """Apply a visual material to the garment mesh."""
        self.visual_material_path = find_unique_string_name(
            self.usd_prim_path + "/Looks/visual_material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )

        add_reference_to_stage(
            usd_path=material_path, prim_path=self.visual_material_path
        )

        self.visual_material_prim = prims_utils.get_prim_at_path(
            self.visual_material_path
        )

        if not self.visual_material_prim or not self.visual_material_prim.IsValid():
            print(f"Warning: Could not get valid prim at {self.visual_material_path}")
            return
        children = prims_utils.get_prim_children(self.visual_material_prim)
        if not children:
            print(
                f"Warning: Material prim at {self.visual_material_path} has no children."
            )
            return

        self.material_prim = children[0]
        self.material_prim_path = self.material_prim.GetPath()
        self.visual_material = PreviewSurface(self.material_prim_path)

        mesh_prim_to_bind = prims_utils.get_prim_at_path(self.mesh_prim_path)
        if not mesh_prim_to_bind:
            print(
                f"Warning: Could not find mesh prim at {self.mesh_prim_path} to bind material."
            )
            return

        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.mesh_prim_path,
            material_path=self.material_prim_path,
            strength=UsdShade.Tokens.strongerThanDescendants,
        )

        garment_submesh = prims_utils.get_prim_children(mesh_prim_to_bind)
        if len(garment_submesh) > 0:
            for prim in garment_submesh:
                if prim.IsA(UsdGeom.Gprim):
                    omni.kit.commands.execute(
                        "BindMaterialCommand",
                        prim_path=prim.GetPath(),
                        material_path=self.material_prim_path,
                        strength=UsdShade.Tokens.strongerThanDescendants,
                    )

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

    def _get_initial_info(self):
        """Capture initial particle positions for reset functionality."""
        if self._device == "cpu":
            self.initial_points_positions = (
                self._get_points_pose().detach().cpu().numpy()
            )
        else:
            self.physics_sim_view = SimulationManager.get_physics_sim_view()
            self._cloth_prim_view.initialize(self.physics_sim_view)
            self.initial_points_positions = self._cloth_prim_view.get_world_positions()

    def transform_points(self, points, pos, ori, scale):
        """
        Transform local points to world space using position, orientation, and scale.

        Args:
            points: (N, 3) array of local points
            pos: (3,) position vector
            ori: (4,) quaternion orientation
            scale: Scale factor (numpy array)

        Returns:
            (N, 3) array of transformed points in world space
        """
        ori_matrix = quat_to_rot_matrix(ori)
        scaled_points = points * scale
        transformed_points = scaled_points @ ori_matrix.T + pos
        return transformed_points

    def inverse_transform_points(self, transformed_points, pos, ori, scale):
        """
        Transform world space points back to local space.

        Args:
            transformed_points: (N, 3) array of world space points
            pos: (3,) position vector
            ori: (4,) quaternion orientation
            scale: Scale factor (numpy array)

        Returns:
            (N, 3) array of points in local space
        """
        ori_matrix = quat_to_rot_matrix(ori)
        shifted_points = transformed_points - pos
        rotated_points = shifted_points @ ori_matrix
        original_points = rotated_points / scale
        return original_points

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

        if "garment_config" in modified_config:
            garment_config = modified_config["garment_config"].copy()
            garment_params_to_randomize = [
                "particle_mass",
                "stretch_stiffness",
                "bend_stiffness",
                "shear_stiffness",
                "spring_damping",
            ]

            for param in garment_params_to_randomize:
                if param in garment_config and garment_config[param] is not None:
                    original_value = garment_config[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        garment_config[param] = random.uniform(min_val, max_val)

            modified_config["garment_config"] = garment_config

        if "particle_system" in modified_config:
            particle_system = modified_config["particle_system"].copy()
            particle_system_params_to_randomize = [
                "solver_position_iteration_count",
                "max_depenetration_velocity",
                "contact_offset",
                "rest_offset",
                "particle_contact_offset",
                "fluid_rest_offset",
                "solid_rest_offset",
                "max_neighborhood",
                "max_velocity",
            ]

            for param in particle_system_params_to_randomize:
                if param in particle_system and particle_system[param] is not None:
                    original_value = particle_system[param]
                    if isinstance(original_value, (int, float)):
                        variation = original_value * (ratio - 1)
                        min_val = original_value - variation
                        max_val = original_value + variation
                        particle_system[param] = random.uniform(min_val, max_val)
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
                        particle_system[param] = randomized_list

            modified_config["particle_system"] = particle_system

        if "particle_material" in modified_config:
            particle_material = modified_config["particle_material"].copy()
            material_params_to_randomize = [
                "adhesion",
                "adhesion_offset_scale",
                "cohesion",
                "particle_adhesion_scale",
                "particle_friction_scale",
                "drag",
                "lift",
                "friction",
                "damping",
                "gravity_scale",
                "viscosity",
                "vorticity_confinement",
                "surface_tension",
            ]

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
        """Get the state of the garment object.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.

        Returns:
            Dictionary containing:
                - root_pose: torch.Tensor, shape (7,), root pose [pos(3), quat(4)]
                - root_velocity: torch.Tensor, shape (6,), root velocity [lin_vel(3), ang_vel(3)]
                - asset_info: dict with usd_path and primitive_type
        """

        try:
            pos_world, ori_world = self.get_world_pose()

            if isinstance(pos_world, np.ndarray):
                pos_tensor = torch.tensor(pos_world, dtype=torch.float32)
            elif isinstance(pos_world, torch.Tensor):
                pos_tensor = pos_world.clone().detach().to(dtype=torch.float32)
            else:
                pos_tensor = torch.tensor(pos_world, dtype=torch.float32)

            if isinstance(ori_world, np.ndarray):
                ori_tensor = torch.tensor(ori_world, dtype=torch.float32)
            elif isinstance(ori_world, torch.Tensor):
                ori_tensor = ori_world.clone().detach().to(dtype=torch.float32)
            else:
                ori_tensor = torch.tensor(ori_world, dtype=torch.float32)

            if pos_tensor.ndim == 0:
                pos_tensor = pos_tensor.unsqueeze(0)
            if pos_tensor.shape[0] < 3:
                pos_tensor = torch.cat(
                    [
                        pos_tensor,
                        torch.zeros(3 - pos_tensor.shape[0], dtype=torch.float32),
                    ]
                )
            pos_tensor = pos_tensor[:3]

            if ori_tensor.ndim == 0:
                ori_tensor = ori_tensor.unsqueeze(0)
            if ori_tensor.shape[0] < 4:
                ori_tensor = torch.cat(
                    [
                        ori_tensor,
                        torch.zeros(4 - ori_tensor.shape[0], dtype=torch.float32),
                    ]
                )
            ori_tensor = ori_tensor[:4]

            root_pose = torch.cat([pos_tensor, ori_tensor])

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
                                dtype=torch.float32,
                            ),
                        ]
                    )
                root_pose[:3] -= env_origin_tensor[:3]
        except (AttributeError, RuntimeError, Exception):
            root_pose = torch.zeros(7, dtype=torch.float32)

        try:
            if hasattr(self, "_cloth_prim_view") and self._cloth_prim_view is not None:
                nodal_velocity = self._cloth_prim_view.get_velocities()
                if not isinstance(nodal_velocity, torch.Tensor):
                    nodal_velocity = torch.tensor(nodal_velocity, dtype=torch.float32)

                if nodal_velocity.ndim == 3 and nodal_velocity.shape[0] == 1:
                    nodal_velocity = nodal_velocity.squeeze(0)
                elif nodal_velocity.ndim == 3 and nodal_velocity.shape[0] > 1:
                    nodal_velocity = nodal_velocity[0]

                if nodal_velocity.numel() > 0 and nodal_velocity.shape[0] > 0:
                    root_lin_vel = torch.mean(nodal_velocity, dim=0)
                    if root_lin_vel.shape[0] < 3:
                        root_lin_vel = torch.cat(
                            [
                                root_lin_vel,
                                torch.zeros(
                                    3 - root_lin_vel.shape[0], dtype=torch.float32
                                ),
                            ]
                        )
                    root_lin_vel = root_lin_vel[:3]
                else:
                    root_lin_vel = torch.zeros(3, dtype=torch.float32)
            else:
                root_lin_vel = torch.zeros(3, dtype=torch.float32)

            root_ang_vel = torch.zeros(3, dtype=torch.float32)

            root_velocity = torch.cat([root_lin_vel, root_ang_vel])
        except (AttributeError, RuntimeError, Exception):
            root_velocity = torch.zeros(6, dtype=torch.float32)

        asset_info = {
            "usd_path": self.usd_path if hasattr(self, "usd_path") else None,
            "primitive_type": self.primitive_type
            if hasattr(self, "primitive_type")
            else None,
        }

        return {
            "root_pose": root_pose,
            "root_velocity": root_velocity,
            "asset_info": asset_info,
        }
