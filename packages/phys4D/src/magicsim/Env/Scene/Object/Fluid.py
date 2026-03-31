import os
import random
import re
import numpy as np
import omni.kit.commands
import carb
from scipy.spatial import Delaunay
import open3d as o3d
import torch
from omni.physx.scripts import particleUtils, physicsUtils
from isaacsim.replicator.behavior.utils.scene_utils import create_mdl_material
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.core.utils.prims import is_prim_path_valid, delete_prim
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.prims import SingleGeometryPrim

from pxr import UsdGeom, Sdf, Gf, Vt, PhysxSchema
from omegaconf import DictConfig


class FluidObject:
    """
    FluidObject class for simulating fluid particles in Isaac Sim.
    Manages fluid particle systems, containers, materials, and physics properties.
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
        Initialize the FluidObject with configuration, USD assets, and physics setup.

        Args:
            prim_path: Path to the prim in the stage
            usd_path: Path to the USD asset file for this fluid
            config: Configuration dictionary containing fluid properties
        """

        carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
        carb.settings.get_settings().set_bool("/physics/updateParticlesToUsd", True)

        self.prim_path = prim_path
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
        self.physics_cfg = self.instance_config.get("physics", {})

        self.physics_cfg = self._apply_physics_ratio_randomization(self.physics_cfg)

        self.layout_manager = layout_manager
        self.layout_info = layout_info
        self.env_origin = env_origin.detach().cpu().numpy()

        self.usd_prim_path = prim_path
        self.usd_path = usd_path
        self.prim_name = self.instance_name
        self.env_name = prim_path_parts[-4]
        self.mesh_prim_path = self.usd_prim_path + "/mesh"
        self.stage = get_current_stage()

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

        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        container_cfg = self.instance_config.get("container", {})
        container_enabled = container_cfg.get("enabled", False)
        container_usd_path = container_cfg.get(
            "usd", "./Assets/Object/Fluid_Container/new_container.usd"
        )
        self.container_offset = container_cfg.get("offset", [0.0, 0.0, 0.0])
        container_scale = container_cfg.get(
            "scale",
            self.instance_config.get("visual", {}).get(
                "container_scale", [1.0, 1.0, 1.0]
            ),
        )

        self.container = None
        self.container_prim_path = None
        self.container_position = None

        if container_enabled and container_usd_path:
            container_path = find_unique_string_name(
                initial_name=os.path.dirname(prim_path) + "/container",
                is_unique_fn=lambda x: not is_prim_path_valid(x),
            )
            self.container_prim_path = container_path
            add_reference_to_stage(
                usd_path=container_usd_path, prim_path=container_path
            )
            self.container_position = Gf.Vec3d(
                float(self.init_pos[0] + self.container_offset[0]),
                float(self.init_pos[1] + self.container_offset[1]),
                float(self.container_offset[2]),
            )

            self.container = SingleGeometryPrim(
                prim_path=container_path,
                name=f"fluid_container_{self.prim_name}",
                collision=True,
                scale=container_scale,
            )

        inst_particle_system_cfg = self.physics_cfg.get("particle_system", {})
        interaction_flag = self.category_config.get("interaction_with_object", False)

        if interaction_flag:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/particle_system"
            )
        else:
            self.particle_system_path = (
                f"/Particle_Attribute/{self.env_name}/fluid_particle_system"
            )

        if not is_prim_path_valid(self.particle_system_path):
            self.particle_system = PhysxSchema.PhysxParticleSystem.Define(
                self.stage, self.particle_system_path
            )
        else:
            prim = self.stage.GetPrimAtPath(self.particle_system_path)
            self.particle_system = PhysxSchema.PhysxParticleSystem(prim)

        self.particle_system.CreateParticleContactOffsetAttr().Set(
            inst_particle_system_cfg.get("particle_contact_offset", 0.025)
        )
        self.particle_system.CreateContactOffsetAttr().Set(
            inst_particle_system_cfg.get("contact_offset", 0.025)
        )
        self.particle_system.CreateRestOffsetAttr().Set(
            inst_particle_system_cfg.get("rest_offset", 0.0225)
        )
        self.particle_system.CreateFluidRestOffsetAttr().Set(
            inst_particle_system_cfg.get("fluid_rest_offset", 0.0135)
        )
        self.particle_system.CreateSolidRestOffsetAttr().Set(
            inst_particle_system_cfg.get("solid_rest_offset", 0.0225)
        )
        self.particle_system.CreateMaxVelocityAttr().Set(
            inst_particle_system_cfg.get("max_velocity", 2.5)
        )

        if inst_particle_system_cfg.get("smoothing", False):
            PhysxSchema.PhysxParticleSmoothingAPI.Apply(self.particle_system.GetPrim())
        if inst_particle_system_cfg.get("anisotropy", False):
            PhysxSchema.PhysxParticleAnisotropyAPI.Apply(self.particle_system.GetPrim())
        if inst_particle_system_cfg.get("isosurface", True):
            PhysxSchema.PhysxParticleIsosurfaceAPI.Apply(self.particle_system.GetPrim())

        fluid_mesh = UsdGeom.Mesh.Get(self.stage, Sdf.Path(self.mesh_prim_path))
        fluid_volume_multiplier = self.physics_cfg.get(
            "fluid_volume", self.physics_cfg.get("fluid_volumn", 1.0)
        )
        cloud_points_base = np.array(fluid_mesh.GetPointsAttr().Get())
        visual_scale = (
            np.array(
                self.instance_config.get("visual", {}).get("scale", [1.0, 1.0, 1.0]),
                dtype=np.float32,
            )
            if isinstance(
                self.instance_config.get("visual", {}).get("scale", [1.0, 1.0, 1.0]),
                (list, tuple, np.ndarray),
            )
            else np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )
        self.visual_scale = visual_scale
        cloud_points = cloud_points_base * fluid_volume_multiplier * self.visual_scale
        fluid_rest_offset = inst_particle_system_cfg.get("fluid_rest_offset", 0.0135)
        particleSpacing = 2.0 * fluid_rest_offset

        self.init_particle_positions, self.init_particle_velocities = (
            generate_particles_in_convex_mesh(
                vertices=cloud_points, sphere_diameter=particleSpacing, visualize=False
            )
        )
        self.stage.GetPrimAtPath(self.mesh_prim_path).SetActive(False)

        self.particle_point_instancer_path = Sdf.Path(self.usd_prim_path).AppendChild(
            "particles"
        )

        particleUtils.add_physx_particleset_pointinstancer(
            stage=self.stage,
            path=self.particle_point_instancer_path,
            positions=Vt.Vec3fArray(self.init_particle_positions),
            velocities=Vt.Vec3fArray(self.init_particle_velocities),
            particle_system_path=self.particle_system_path,
            self_collision=True,
            fluid=True,
            particle_group=0,
            particle_mass=0.001,
            density=0.0,
        )

        self.point_instancer = UsdGeom.PointInstancer.Get(
            self.stage, self.particle_point_instancer_path
        )

        init_scale_array = np.array(self.init_scale, dtype=np.float32)
        combined_scale = init_scale_array * self.visual_scale

        physicsUtils.set_or_add_scale_orient_translate(
            self.point_instancer,
            translate=Gf.Vec3f([float(v) for v in self.init_pos]),
            orient=Gf.Quatf(
                float(self.init_ori[0]),
                Gf.Vec3f(
                    float(self.init_ori[1]),
                    float(self.init_ori[2]),
                    float(self.init_ori[3]),
                ),
            ),
            scale=Gf.Vec3f([float(v) for v in combined_scale]),
        )

        proto_path = self.particle_point_instancer_path.AppendChild(
            "particlePrototype0"
        )
        self.particle_prototype_path = proto_path

        particle_prototype_sphere = UsdGeom.Sphere.Get(self.stage, proto_path)
        particle_prototype_sphere.CreateRadiusAttr().Set(fluid_rest_offset)
        if inst_particle_system_cfg.get("isosurface", True):
            UsdGeom.Imageable(particle_prototype_sphere).MakeInvisible()

        self._apply_random_material()

        self._handle_semantic_labels()

    def _apply_random_material(self):
        """
        Selects a random material from configuration, creates it, and binds it
        to the particle system and particle prototypes. Also sets physics properties.
        """
        visual_cfg = self.instance_config.get("visual", {})
        material_cfg = visual_cfg.get("visual_material", {})
        material_list = material_cfg.get("material_usd_folder", [])

        if material_list:
            material_url = random.choice(material_list)
        else:
            material_url = "./Assets/Material/Base/Textiles/Linen_Blue.mdl"

        material_name = os.path.splitext(os.path.basename(material_url))[0]
        looks_path = f"{os.path.dirname(self.prim_path)}/Looks"
        if is_prim_path_valid(f"{looks_path}/material"):
            delete_prim(f"{looks_path}/material")

        unique_material_name = find_unique_string_name(
            initial_name=f"{looks_path}/material",
            is_unique_fn=lambda x: not is_prim_path_valid(x),
        )
        color_material_path = unique_material_name
        create_mdl_material(material_url, material_name, color_material_path)

        inst_particle_material_cfg = self.physics_cfg.get("particle_material", {})
        particleUtils.add_pbd_particle_material(
            stage=self.stage,
            path=color_material_path,
            adhesion=inst_particle_material_cfg.get("adhesion"),
            adhesion_offset_scale=inst_particle_material_cfg.get(
                "adhesion_offset_scale"
            ),
            cohesion=inst_particle_material_cfg.get("cohesion"),
            particle_adhesion_scale=inst_particle_material_cfg.get(
                "particle_adhesion_scale"
            ),
            particle_friction_scale=inst_particle_material_cfg.get(
                "particle_friction_scale"
            ),
            drag=inst_particle_material_cfg.get("drag"),
            lift=inst_particle_material_cfg.get("lift"),
            friction=inst_particle_material_cfg.get("friction"),
            damping=inst_particle_material_cfg.get("damping"),
            gravity_scale=inst_particle_material_cfg.get("gravity_scale", 1.0),
            viscosity=inst_particle_material_cfg.get("viscosity"),
            vorticity_confinement=inst_particle_material_cfg.get(
                "vorticity_confinement"
            ),
            surface_tension=inst_particle_material_cfg.get("surface_tension"),
            density=inst_particle_material_cfg.get("density"),
            cfl_coefficient=inst_particle_material_cfg.get("cfl_coefficient"),
        )

        omni.kit.commands.execute(
            "BindMaterialCommand",
            prim_path=self.particle_system_path,
            material_path=color_material_path,
        )

        if hasattr(self, "particle_prototype_path") and is_prim_path_valid(
            self.particle_prototype_path
        ):
            omni.kit.commands.execute(
                "BindMaterialCommand",
                prim_path=self.particle_prototype_path,
                material_path=color_material_path,
            )

    def _re_instance_name(self, inst_name):
        """Reformats the instance name to ensure consistent numbering."""
        parts = inst_name.split("_")
        cat_name_extracted = "_".join(parts[:-1])
        obj_id_str = parts[-1]
        obj_id = int(obj_id_str)
        original_id = (obj_id - 1) % self.num_per_env + 1
        return f"{cat_name_extracted}_{original_id}"

    def initialize(self):
        """Initialize the fluid container position."""

        if (
            hasattr(self, "container")
            and self.container
            and self.container_position is not None
        ):
            self.container.set_local_pose(translation=self.container_position)
        else:
            print(
                f"Warning: Container not initialized for {self.prim_path}, cannot set initial pose."
            )

    def reset(self, soft=False):
        """
        Reset the fluid system to initial state with new position and orientation.

        Args:
            soft: If True, use soft reset ranges; otherwise use initial ranges
        """
        self._apply_random_material()

        if not self.layout_manager:
            raise RuntimeError(
                f"LayoutManager is required for {self.prim_path}. All position information must come from LayoutManager."
            )

        env_id = self._extract_env_id_from_prim_path()
        if env_id is None:
            print(
                f"Warning: Could not extract env_id for {self.prim_path}. Cannot perform reset."
            )
            return

        reset_type = "soft" if soft else "hard"
        new_layout = self.layout_manager.generate_new_layout(
            env_id=env_id, prim_path=self.usd_prim_path, reset_type=reset_type
        )

        if not new_layout:
            print(
                f"Warning: LayoutManager did not provide new layout for {self.prim_path}. Cannot perform reset."
            )
            return

        pos = new_layout["pos"]
        ori_quat = new_layout["ori"]

        if self.container is not None:
            self.container_position = Gf.Vec3d(
                float(pos[0] + self.container_offset[0]),
                float(pos[1] + self.container_offset[1]),
                float(self.container_offset[2]),
            )

        if (
            hasattr(self, "container")
            and self.container
            and self.container_position is not None
        ):
            self.container.set_local_pose(translation=self.container_position)
        else:
            print(
                f"Warning: Container not initialized for {self.prim_path}, cannot reset pose."
            )

        self.set_particle_positions(self.init_particle_positions)

        physicsUtils.set_or_add_translate_op(
            self.point_instancer, translate=Gf.Vec3f([float(v) for v in pos])
        )

        physicsUtils.set_or_add_orient_op(
            self.point_instancer,
            orient=Gf.Quatf(
                float(ori_quat[0]),
                Gf.Vec3f(float(ori_quat[1]), float(ori_quat[2]), float(ori_quat[3])),
            ),
        )

    def get_particle_positions(self, visualize: bool = True):
        """
        Get current positions of all fluid particles.

        Args:
            visualize: Whether to visualize particles using Open3D

        Returns:
            positions: Array of particle positions
        """

        if not hasattr(self, "point_instancer") or not self.point_instancer:
            print(
                f"Warning: point_instancer not valid for {self.prim_path}. Cannot get positions."
            )
            return np.array([]), None, None

        positions_attr = self.point_instancer.GetPositionsAttr()
        if not positions_attr:
            print(
                f"Warning: Could not get PositionsAttr for {self.particle_point_instancer_path}."
            )
            return np.array([]), None, None

        positions = np.array(positions_attr.Get(), dtype=np.float32)

        if visualize:
            if positions.size == 0:
                print("Warning: No particle positions to visualize.")
                return positions, None, None
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(positions)
                o3d.visualization.draw_geometries([pcd])
            except Exception as e:
                print(f"Error during visualization: {e}")

        return positions, None, None

    def set_particle_positions(self, positions: np.ndarray):
        """
        Set positions of all fluid particles.

        Args:
            positions: Array of new particle positions
        """

        if not hasattr(self, "point_instancer") or not self.point_instancer:
            print(
                f"Warning: point_instancer not valid for {self.prim_path}. Cannot set positions."
            )
            return

        if not isinstance(positions, np.ndarray):
            try:
                if positions and isinstance(positions[0], Gf.Vec3f):
                    positions = np.array(
                        [[p[0], p[1], p[2]] for p in positions], dtype=np.float32
                    )
                else:
                    positions = np.array(positions, dtype=np.float32)
            except Exception as e:
                print(
                    f"Error converting positions to numpy array: {e}. Positions type: {type(positions)}"
                )
                return

        if positions.ndim != 2 or positions.shape[1] != 3:
            print(
                f"Error: Invalid shape for positions array: {positions.shape}. Expected (N, 3)."
            )
            return

        try:
            positions_vt = Vt.Vec3fArray.FromNumpy(positions.astype(np.float32))
        except Exception as e:
            print(f"Error converting numpy array to Vt.Vec3fArray: {e}")
            return

        positions_attr = self.point_instancer.GetPositionsAttr()
        if not positions_attr:
            print(
                f"Warning: Could not get PositionsAttr for {self.particle_point_instancer_path} to set positions."
            )
            return

        positions_attr.Set(positions_vt)

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
            "fluid_volumn",
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

        if "particle_system" in modified_config:
            particle_system = modified_config["particle_system"].copy()
            particle_system_params_to_randomize = [
                "particle_contact_offset",
                "contact_offset",
                "rest_offset",
                "fluid_rest_offset",
                "solid_rest_offset",
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
                "density",
                "cfl_coefficient",
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

    def _handle_semantic_labels(self):
        """Manage semantic labeling: clear existing labels and apply new ones."""

        prim = self.stage.GetPrimAtPath(self.particle_point_instancer_path)
        if prim and prim.IsValid():
            remove_labels(prim, include_descendants=True)
            semantic_label = self._get_semantic_label()
            if semantic_label:
                add_labels(prim, [semantic_label])
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

    def get_state(self, is_relative: bool = False) -> dict[str, torch.Tensor]:
        """Get the state of the fluid object.

        Args:
            is_relative: If True, positions are relative to environment origin. Defaults to False.

        Returns:
            Dictionary containing:
                - particle_positions: torch.Tensor, shape (num_particles, 3), particle positions
                - particle_velocities: torch.Tensor, shape (num_particles, 3), particle velocities
                - asset_info: dict with usd_path and primitive_type
        """
        try:
            positions, _, _ = self.get_particle_positions(visualize=False)
            if positions is None or (
                isinstance(positions, np.ndarray) and positions.size == 0
            ):
                positions = torch.zeros(0, 3, dtype=torch.float32)
            elif not isinstance(positions, torch.Tensor):
                positions = torch.tensor(positions, dtype=torch.float32)
            if positions.dim() == 1 and positions.shape[0] == 3:
                positions = positions.unsqueeze(0)
            elif positions.dim() == 0 or positions.numel() == 0:
                positions = torch.zeros(0, 3, dtype=torch.float32)
        except (AttributeError, RuntimeError):
            positions = torch.zeros(0, 3, dtype=torch.float32)

        if is_relative and hasattr(self, "env_origin") and positions.numel() > 0:
            env_origin_tensor = (
                torch.tensor(
                    self.env_origin, dtype=torch.float32, device=positions.device
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
            positions[:, :3] -= env_origin_tensor[:3]

        try:
            velocities = self.get_particle_velocities()
            if not isinstance(velocities, torch.Tensor):
                velocities = torch.tensor(
                    velocities, dtype=torch.float32, device=positions.device
                )
            if velocities.dim() == 1 and velocities.shape[0] == 3:
                velocities = velocities.unsqueeze(0)
            elif velocities.dim() == 0 or velocities.numel() == 0:
                velocities = torch.zeros_like(positions)
        except (AttributeError, RuntimeError):
            velocities = torch.zeros_like(positions)

        asset_info = {
            "usd_path": self.usd_path if hasattr(self, "usd_path") else None,
            "primitive_type": None,
        }

        return {
            "particle_positions": positions,
            "particle_velocities": velocities,
            "asset_info": asset_info,
        }


def generate_particles_in_convex_mesh(
    vertices: np.ndarray, sphere_diameter: float, visualize: bool = False
):
    """
    Generate particles within a convex mesh using Delaunay triangulation.

    Args:
        vertices: Vertices of the convex mesh
        sphere_diameter: Diameter of particles to generate
        visualize: Whether to visualize the particles and mesh vertices

    Returns:
        List of particle positions and velocities (zero-initialized)
    """

    if not isinstance(vertices, np.ndarray):
        vertices = np.array(vertices)

    if vertices.shape[0] < 4:
        print(
            "Warning: Need at least 4 vertices for Delaunay triangulation. Returning empty."
        )
        return [], []

    try:
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)

        if np.linalg.matrix_rank(vertices) < 3:
            vertices += np.random.rand(*vertices.shape) * 1e-6

        hull = Delaunay(vertices)
    except Exception as e:
        print(
            f"Error during Delaunay triangulation: {e}. Vertices shape: {vertices.shape}. Returning empty."
        )
        return [], []

    epsilon = sphere_diameter * 0.01
    x_vals = np.arange(min_bound[0], max_bound[0] + epsilon, sphere_diameter)
    y_vals = np.arange(min_bound[1], max_bound[1] + epsilon, sphere_diameter)
    z_vals = np.arange(min_bound[2], max_bound[2] + epsilon, sphere_diameter)

    if x_vals.size == 0 or y_vals.size == 0 or z_vals.size == 0:
        print("Warning: Empty dimension range for particle grid. Returning empty.")
        return [], []

    samples = np.stack(
        np.meshgrid(x_vals, y_vals, z_vals, indexing="ij"), axis=-1
    ).reshape(-1, 3)

    inside_mask = hull.find_simplex(samples, tol=1e-6) >= 0
    inside_points = samples[inside_mask]

    velocity = np.zeros_like(inside_points)

    if visualize:
        if inside_points.size == 0:
            print("Warning: No inside points found to visualize.")
        else:
            try:
                particle_pcd = o3d.geometry.PointCloud()
                particle_pcd.points = o3d.utility.Vector3dVector(inside_points)
                particle_pcd.paint_uniform_color([0.2, 0.4, 1.0])

                vertex_pcd = o3d.geometry.PointCloud()
                vertex_pcd.points = o3d.utility.Vector3dVector(vertices)
                vertex_pcd.paint_uniform_color([1.0, 0.1, 0.1])

                o3d.visualization.draw_geometries(
                    [particle_pcd, vertex_pcd],
                    window_name="Convex Mesh Particle Filling",
                )
            except Exception as e:
                print(f"Error during visualization: {e}")

    positions_gf = [
        Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in inside_points
    ]
    velocities_gf = [Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in velocity]

    return positions_gf, velocities_gf
