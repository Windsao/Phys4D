











from copy import deepcopy


import numpy as np
import torch
from nvblox_torch.datasets.mesh_dataset import MeshDataset
from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw
from tqdm import tqdm


from curobo.geom.sdf.world import WorldConfig
from curobo.util.logger import setup_curobo_logger

torch.manual_seed(0)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

np.random.seed(0)


def generate_images():


    file_paths = [demo_raw, motion_benchmaker_raw, mpinets_raw][1:2]

    for file_path in file_paths:
        problems = file_path()

        for key, v in tqdm(problems.items()):
            scene_problems = problems[key]
            i = -1
            for problem in tqdm(scene_problems, leave=False):
                i += 1

                world = WorldConfig.from_dict(deepcopy(problem["obstacles"])).get_mesh_world(
                    merge_meshes=True
                )
                mesh = world.mesh[0].get_trimesh_mesh()


                save_path = "benchmark/log/nvblox/" + key + "_" + str(i)


                MeshDataset(
                    None, n_frames=1, image_size=640, save_data_dir=save_path, trimesh_mesh=mesh
                )


if __name__ == "__main__":
    setup_curobo_logger("error")
    generate_images()
