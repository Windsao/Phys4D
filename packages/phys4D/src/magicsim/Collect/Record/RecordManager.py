from typing import Any, Dict, List
from omegaconf import DictConfig
import torch
import os
import numpy as np
from magicsim.StardardEnv.Robot.TaskBaseEnv import TaskBaseEnv
from magicsim.Env.Utils.file import Logger
from magicsim.Collect.Record.CameraWriter import write_annotator_step
from omni.replicator.core.scripts import functional as F
from omni.replicator.core.scripts.backends import BackendDispatch

OUTPUT_PATH = "/home/user/magicsim/MagicSim/TestOutput"
JSON_INDENT = 4


class RecordManager:
    def __init__(
        self,
        env: TaskBaseEnv,
        num_envs: int,
        record_config: DictConfig,
        device: torch.device,
        logger: Logger,
    ):
        self.env = env
        self.num_envs = num_envs
        self.record_config = record_config
        self.device = device
        self.logger = logger
        self.record_obs_buffer: List[List[Dict[str, Any]]] = []
        self.record_action_buffer: List[List[Dict[str, Any]]] = []
        self.record_collect_buffer: List[List[Dict[str, Any]]] = []

        self._saved_for_completed_task: List[bool] = [False] * num_envs

        self.output_path = OUTPUT_PATH
        try:
            if hasattr(record_config, "output_dir") and record_config.output_dir:
                self.output_path = str(record_config.output_dir)
        except Exception:
            self.output_path = OUTPUT_PATH

        self.max_trajectories = 1000
        try:
            if (
                hasattr(record_config, "max_trajectories")
                and record_config.max_trajectories
            ):
                self.max_trajectories = int(record_config.max_trajectories)
        except Exception:
            pass

        self.trajectory_id = self._get_next_trajectory_id()

        self._reached_trajectory_limit = self.trajectory_id >= self.max_trajectories

        if self._reached_trajectory_limit:
            self.logger.warning(
                f"Trajectory limit ({self.max_trajectories}) already reached. "
                f"Current trajectory_id: {self.trajectory_id}. No new trajectories will be saved."
            )
        else:
            self.logger.info(
                f"Initialized RecordManager with trajectory_id starting from {self.trajectory_id}, max_trajectories: {self.max_trajectories}"
            )

    def _get_next_trajectory_id(self) -> int:
        """Find the next available trajectory ID by scanning existing trajectory directories.

        Returns:
            The next available trajectory ID (will be 0 if no trajectories exist,
            or max(existing_ids) + 1 if trajectories exist).
        """
        if not os.path.exists(self.output_path):
            return 0

        existing_ids = []
        try:
            for entry in os.listdir(self.output_path):
                entry_path = os.path.join(self.output_path, entry)

                if os.path.isdir(entry_path):
                    try:
                        trajectory_id = int(entry)
                        existing_ids.append(trajectory_id)
                    except ValueError:
                        continue
        except Exception as e:
            self.logger.warning(
                f"Error scanning existing trajectories: {e}. Starting from 0."
            )
            return 0

        if not existing_ids:
            return 0

        next_id = max(existing_ids) + 1
        self.logger.info(
            f"Found existing trajectories: {sorted(existing_ids)}. Next trajectory ID: {next_id}"
        )
        return next_id

    @staticmethod
    def _to_serializable(value):
        """Convert data (including torch / numpy) to JSON-serializable types."""
        if hasattr(value, "cpu"):
            if isinstance(value, torch.Tensor):
                assert not torch.isnan(value).any(), (
                    "Found nan values in tensor during serialization. "
                    "This indicates a bug in padding or data processing."
                )
            value = value.cpu().numpy()
        if isinstance(value, np.ndarray):
            assert not np.isnan(value).any(), (
                "Found nan values in numpy array during serialization. "
                "This indicates a bug in padding or data processing."
            )

            result = value.tolist()

            return RecordManager._to_serializable(result)
        if isinstance(value, dict):
            return {k: RecordManager._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [RecordManager._to_serializable(v) for v in value]

        if isinstance(value, (float, np.floating, np.integer)):
            if isinstance(value, (float, np.floating)):
                assert not np.isnan(value), (
                    "Found nan float value during serialization. "
                    "This indicates a bug in padding or data processing."
                )

            if isinstance(value, (float, np.floating)):
                return round(float(value), 4)
            else:
                return int(value)
        return value

    def _convert_batched_env_info_to_per_env(
        self, batched_env_info: Dict[str, Any], env_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Convert batched env_info format to per-env format.

        env_info should contain: obs, reward, terminated, truncated, info
        where obs contains: policy_obs, privilege_obs, camera_info

        Args:
            batched_env_info: Batched format where camera_info is organized by cam_id
            env_ids: List of environment IDs to process

        Returns:
            Dictionary mapping env_id to env_info dictionary
        """

        state_backup = None
        if "info" in batched_env_info and isinstance(batched_env_info["info"], dict):
            state_backup = batched_env_info["info"].pop("state", None)

        self._check_dict_values_length(
            batched_env_info, self.num_envs, "batched_env_info"
        )

        if state_backup is not None and "info" in batched_env_info:
            batched_env_info["info"]["state"] = state_backup

        if (
            "terminated" in batched_env_info
            and batched_env_info["terminated"] is not None
        ):
            terminated = batched_env_info["terminated"]
            if isinstance(terminated, torch.Tensor):
                assert len(terminated) == self.num_envs, (
                    f"terminated has length {len(terminated)}, expected {self.num_envs}"
                )
        if (
            "truncated" in batched_env_info
            and batched_env_info["truncated"] is not None
        ):
            truncated = batched_env_info["truncated"]
            if isinstance(truncated, torch.Tensor):
                assert len(truncated) == self.num_envs, (
                    f"truncated has length {len(truncated)}, expected {self.num_envs}"
                )

        per_env_info = {}

        if len(batched_env_info) == 5:
            obs = batched_env_info["obs"]
            reward = batched_env_info["reward"]
            terminated = batched_env_info["terminated"]
            truncated = batched_env_info["truncated"]
            info = batched_env_info["info"]
        else:
            obs = batched_env_info["obs"]
            info = batched_env_info["info"]
            reward = None
            terminated = None
            truncated = None

        policy_obs = obs.get("policy_obs", {})
        camera_info = policy_obs.get("camera_info", [])

        action_info = policy_obs.pop("last_action", None)
        camera_action_info = policy_obs.pop("last_camera_action", None)
        if action_info is None and camera_action_info is not None:
            action_info = camera_action_info
        privilege_obs = obs.get("privilege_obs", {})

        if len(batched_env_info) == 5:
            assert action_info is not None, "action_info should not be None"

        env_camera_info = {}
        if camera_info:
            num_cams = len(camera_info)
            if num_cams > 0:
                annotator_names = list(camera_info[0].keys())

                for env_id in env_ids:
                    env_cam_data = {}
                    for annotator_name in annotator_names:
                        cam_list = []
                        for cam_id in range(num_cams):
                            if annotator_name in camera_info[cam_id]:
                                annotator_data = camera_info[cam_id][annotator_name]

                                if isinstance(annotator_data, (list, torch.Tensor)):
                                    if env_id < len(annotator_data):
                                        cam_list.append(annotator_data[env_id])
                                    else:
                                        cam_list.append(None)
                                else:
                                    cam_list.append(annotator_data)
                            else:
                                cam_list.append(None)
                        env_cam_data[annotator_name] = cam_list

                    if "camera_params" in env_cam_data:
                        env_cam_data["camera_params"] = (
                            self._add_local_pose_to_camera_params_at_record_time(
                                env_cam_data["camera_params"], env_id
                            )
                        )

                    env_camera_info[env_id] = env_cam_data

        for env_id in env_ids:
            env_policy_obs = {}

            if "robot_state" in policy_obs:
                env_robot_state = {}
                for robot_name, robot_data in policy_obs["robot_state"][0].items():
                    env_robot_state[robot_name] = {}
                    for key, value in robot_data.items():
                        if isinstance(value, torch.Tensor):
                            env_robot_state[robot_name][key] = value[env_id]
                        else:
                            env_robot_state[robot_name][key] = value
                env_policy_obs["robot_state"] = [env_robot_state]

            if env_id in env_camera_info:
                env_policy_obs["camera_info"] = env_camera_info[env_id]

            if action_info:
                env_action_info = {}
                for key, value in action_info.items():
                    if isinstance(value, torch.Tensor):
                        env_value = value[env_id]
                        if env_value.ndim > 0:
                            assert not torch.isnan(env_value).any(), (
                                f"Found nan in action_info['{key}'][{env_id}]. This indicates a bug in padding."
                            )
                        else:
                            assert not torch.isnan(env_value), (
                                f"Found nan in action_info['{key}'][{env_id}]. This indicates a bug in padding."
                            )
                        env_action_info[key] = env_value
                    elif isinstance(value, dict):
                        env_action_info[key] = {}
                        for robot_name, robot_actions in value.items():
                            if isinstance(robot_actions, torch.Tensor):
                                env_action_info[key][robot_name] = robot_actions[env_id]
                                continue

                            env_action_info[key][robot_name] = {}
                            for term_name, term_actions in robot_actions.items():
                                if isinstance(term_actions, torch.Tensor):
                                    term_value = term_actions[env_id]
                                    if term_value.ndim > 0:
                                        assert not torch.isnan(term_value).any(), (
                                            f"Found nan in action_info['{key}']['{robot_name}']['{term_name}'][{env_id}]. "
                                            f"This indicates a bug in padding."
                                        )
                                    else:
                                        assert not torch.isnan(term_value), (
                                            f"Found nan in action_info['{key}']['{robot_name}']['{term_name}'][{env_id}]. "
                                            f"This indicates a bug in padding."
                                        )

                                    env_action_info[key][robot_name][term_name] = (
                                        term_value
                                    )
                                else:
                                    env_action_info[key][robot_name][term_name] = (
                                        term_actions
                                    )
                    else:
                        env_action_info[key] = value
                env_policy_obs["last_action"] = env_action_info

            if camera_action_info:
                env_camera_action_info: Dict[str, Any] = {}
                for key, value in camera_action_info.items():
                    if isinstance(value, torch.Tensor):
                        env_camera_action_info[key] = value[env_id]
                    elif isinstance(value, dict):
                        env_camera_action_info[key] = self._extract_env_from_dict(
                            value, env_id
                        )
                    else:
                        env_camera_action_info[key] = value
                env_policy_obs["last_camera_action"] = env_camera_action_info

            for key, value in policy_obs.items():
                if key in ["robot_state", "camera_info", "last_action"]:
                    continue

                if isinstance(value, torch.Tensor):
                    env_policy_obs[key] = value[env_id]
                elif isinstance(value, dict):
                    env_policy_obs[key] = self._extract_env_from_dict(value, env_id)
                else:
                    env_policy_obs[key] = value

            env_privilege_obs = {}
            for key, value in privilege_obs.items():
                if isinstance(value, torch.Tensor):
                    env_privilege_obs[key] = value[env_id]
                elif isinstance(value, dict):
                    env_privilege_obs[key] = self._extract_env_from_dict(value, env_id)
                else:
                    env_privilege_obs[key] = value

            env_obs = {
                "policy_obs": env_policy_obs,
                "privilege_obs": env_privilege_obs,
            }

            env_reward = (
                reward[env_id]
                if reward is not None and isinstance(reward, torch.Tensor)
                else None
            )
            env_terminated = (
                terminated[env_id]
                if terminated is not None and isinstance(terminated, torch.Tensor)
                else None
            )
            env_truncated = (
                truncated[env_id]
                if truncated is not None and isinstance(truncated, torch.Tensor)
                else None
            )

            env_info_dict = {}
            if info:
                if isinstance(info, list):
                    if env_id < len(info):
                        env_info_dict = info[env_id] if info[env_id] is not None else {}
                    else:
                        env_info_dict = {}

                elif isinstance(info, dict):
                    for key, value in info.items():
                        if key == "state":
                            if isinstance(value, dict):
                                env_info_dict[key] = self._extract_env_from_dict(
                                    value, env_id
                                )
                            elif isinstance(value, list):
                                if env_id < len(value):
                                    env_info_dict[key] = value[env_id]
                                else:
                                    env_info_dict[key] = {}
                            else:
                                env_info_dict[key] = value
                            continue

                        if isinstance(value, torch.Tensor):
                            env_info_dict[key] = value[env_id]
                        elif isinstance(value, dict):
                            env_info_dict[key] = self._extract_env_from_dict(
                                value, env_id
                            )
                        else:
                            env_info_dict[key] = value
                else:
                    env_info_dict = {}

            per_env_info[env_id] = {
                "obs": env_obs,
                "reward": env_reward,
                "terminated": env_terminated,
                "truncated": env_truncated,
                "info": env_info_dict,
            }

        return per_env_info

    def _extract_env_from_dict(
        self, data: Dict[str, Any], env_id: int
    ) -> Dict[str, Any]:
        """Recursively extract single env data from batched dict."""
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                result[key] = value[env_id]
            elif isinstance(value, dict):
                result[key] = self._extract_env_from_dict(value, env_id)
            elif isinstance(value, list):
                per_env_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        per_env_list.append(item[env_id])
                    elif isinstance(item, dict):
                        per_env_list.append(self._extract_env_from_dict(item, env_id))
                    else:
                        per_env_list.append(item)
                result[key] = per_env_list
            else:
                result[key] = value
        return result

    def reset(self, info: Dict[str, Any]):
        batched_env_info = info["env_info"]

        assert isinstance(batched_env_info, tuple), "env_info must be a tuple"
        assert len(batched_env_info) == 2, "env_info must be a tuple of length 2"

        obs_dict, info_list = batched_env_info

        batched_env_info = {
            "obs": obs_dict,
            "info": info_list,
        }

        env_ids = list(range(self.num_envs))
        per_env_info = self._convert_batched_env_info_to_per_env(
            batched_env_info, env_ids
        )

        for env_id in env_ids:
            if len(self.record_obs_buffer) <= env_id:
                self.record_obs_buffer.append([])
            if len(self.record_action_buffer) <= env_id:
                self.record_action_buffer.append([])
            if len(self.record_collect_buffer) <= env_id:
                self.record_collect_buffer.append([])
            self.record_obs_buffer[env_id].append(per_env_info[env_id])

    def _convert_collect_info_to_per_env(
        self, collect_info: Dict[str, Any], env_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Convert collect_info to per-env format.
        collect_info is a dict of {key: value}
        value is a dict of {env_id: value}
        return a list of {key: value}
        """
        per_env_info = {}
        for env_id in env_ids:
            cur_env_info = {}
            for key, value in collect_info.items():
                cur_env_info[key] = value[env_id]
            per_env_info[env_id] = cur_env_info
        return per_env_info

    def step(self, info: Dict[str, Any], ready_env_ids: List[int]):
        batched_env_info = info["env_info"]
        collect_info = info.copy()
        collect_info.pop("env_info")
        assert isinstance(batched_env_info, tuple), "env_info must be a tuple"
        assert len(batched_env_info) == 5, "env_info must be a tuple of length 5"
        obs_dict, reward_tensor, terminated_tensor, truncated_tensor, info_dict = (
            batched_env_info
        )

        batched_env_info = {
            "obs": obs_dict,
            "reward": reward_tensor,
            "terminated": terminated_tensor,
            "truncated": truncated_tensor,
            "info": info_dict,
        }
        per_env_info = self._convert_batched_env_info_to_per_env(
            batched_env_info, ready_env_ids
        )
        per_collect_info = self._convert_collect_info_to_per_env(
            collect_info, ready_env_ids
        )
        for env_id in ready_env_ids:
            assert len(self.record_obs_buffer[env_id]) - 1 == len(
                self.record_action_buffer[env_id]
            ), (
                f"Record Obs Buffer length{len(self.record_obs_buffer[env_id])}, Action Buffer length{len(self.record_action_buffer[env_id])}"
            )
            assert len(self.record_obs_buffer[env_id]) - 1 == len(
                self.record_collect_buffer[env_id]
            ), (
                f"Record Obs Buffer length{len(self.record_obs_buffer[env_id])}, Collect Buffer length{len(self.record_collect_buffer[env_id])}"
            )
            self.record_obs_buffer[env_id].append(per_env_info[env_id])

            action_data = per_env_info[env_id]["obs"]["policy_obs"].get(
                "last_action", {}
            )

            self.record_action_buffer[env_id].append(action_data)
            self.record_collect_buffer[env_id].append(per_collect_info[env_id])

    def update(self, info: Dict[str, Any]):
        for env_id in range(self.num_envs):
            if (
                info["auto_collect_info"][env_id]["state"].split(":")[0] == "success"
                and info["auto_collect_info"][env_id]["finished"]
            ):
                if not self._saved_for_completed_task[env_id]:
                    self.save_to_disk(env_id)
                    new_obs = self.record_obs_buffer[env_id][-1]
                    self.record_obs_buffer[env_id] = []
                    self.record_obs_buffer[env_id].append(new_obs)
                    self.record_action_buffer[env_id] = []
                    self.record_collect_buffer[env_id] = []

                    self._saved_for_completed_task[env_id] = True

                    try:
                        if hasattr(self.env, "scene") and hasattr(
                            self.env.scene, "reset_idx"
                        ):
                            self.env.scene.reset_idx(env_ids=[env_id])
                            self.logger.info(
                                f"Reset env {env_id} after saving trajectory"
                            )
                        elif hasattr(self.env, "reset_idx"):
                            self.env.reset_idx(env_ids=[env_id])
                            self.logger.info(
                                f"Reset env {env_id} after saving trajectory"
                            )

                        self._saved_for_completed_task[env_id] = False
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to reset env {env_id} after saving: {e}"
                        )
            elif (
                info["auto_collect_info"][env_id]["state"].split(":")[0] == "truncated"
                or info["auto_collect_info"][env_id]["state"].split(":")[0] == "failed"
                and not info["auto_collect_info"][env_id]["finished"]
            ):
                print(f"Env {env_id} failed, We will flush our buffer now")
                new_obs = self.record_obs_buffer[env_id][-1]
                self.record_obs_buffer[env_id] = []
                self.record_obs_buffer[env_id].append(new_obs)
                self.record_action_buffer[env_id] = []
                self.record_collect_buffer[env_id] = []

                self._saved_for_completed_task[env_id] = False
            else:
                if info["auto_collect_info"][env_id] is not None and not info[
                    "auto_collect_info"
                ][env_id].get("finished", False):
                    self._saved_for_completed_task[env_id] = False

    def save_to_disk(self, env_id: int):
        """Save trajectory data to disk.

        Creates a folder structure:
        {OUTPUT_PATH}/{trajectory_id}/
            action/
            collect/
        """

        if self._reached_trajectory_limit:
            self.logger.warning(
                f"Trajectory limit ({self.max_trajectories}) reached. Skipping save for env {env_id}."
            )
            return

        if self.trajectory_id >= self.max_trajectories:
            self.logger.warning(
                f"Trajectory limit ({self.max_trajectories}) reached. Current trajectory_id: {self.trajectory_id}. "
                f"Skipping save for env {env_id}."
            )
            self._reached_trajectory_limit = True
            return

        action_buffer = self.record_action_buffer[env_id]
        collect_buffer = self.record_collect_buffer[env_id]
        obs_buffer = self.record_obs_buffer[env_id]

        self.logger.info(
            f"Buffers for env {env_id}: action={len(action_buffer)}, "
            f"collect={len(collect_buffer)}, obs={len(obs_buffer)}"
        )

        assert len(action_buffer) == len(collect_buffer), (
            f"Action buffer length {len(action_buffer)}, collect buffer length {len(collect_buffer)}"
        )
        assert len(action_buffer) == len(obs_buffer) - 1, (
            f"Action buffer length {len(action_buffer)}, obs buffer length {len(obs_buffer)}"
        )

        if len(action_buffer) == 0:
            self.logger.warning(f"No action data to save for env {env_id}")
            return

        trajectory_id = self.trajectory_id
        trajectory_dir = os.path.join(self.output_path, str(trajectory_id))

        self.trajectory_id += 1

        if self.trajectory_id >= self.max_trajectories:
            self.logger.warning(
                f"Trajectory limit ({self.max_trajectories}) will be reached. "
                f"Current trajectory {trajectory_id} saved. Will stop saving future trajectories."
            )
            self._reached_trajectory_limit = True

        action_dir = os.path.join(trajectory_dir, "action")
        collect_dir = os.path.join(trajectory_dir, "collect")
        env_dir = os.path.join(trajectory_dir, "env")
        info_dir = os.path.join(env_dir, "info")
        camera_dir = os.path.join(env_dir, "camera")
        os.makedirs(action_dir, exist_ok=True)
        os.makedirs(collect_dir, exist_ok=True)
        os.makedirs(info_dir, exist_ok=True)
        os.makedirs(camera_dir, exist_ok=True)

        backend = BackendDispatch(output_dir=trajectory_dir)

        for step_idx, action_data in enumerate(action_buffer):
            serializable_action = self._to_serializable(action_data)
            rel_path = os.path.join("action", f"action_{step_idx:04d}.json")
            backend.schedule(
                F.write_json,
                data=serializable_action,
                path=rel_path,
                indent=JSON_INDENT,
            )

        try:
            self.logger.info(
                f"Saving {len(collect_buffer)} collect data entries for env {env_id}"
            )
            for step_idx, collect_data in enumerate(collect_buffer):
                try:
                    serializable_collect = self._to_serializable(collect_data)
                    rel_path = os.path.join("collect", f"collect_{step_idx:04d}.json")
                    backend.schedule(
                        F.write_json,
                        data=serializable_collect,
                        path=rel_path,
                        indent=JSON_INDENT,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Error saving collect data at step {step_idx} for env {env_id}: {e}"
                    )
                    import traceback

                    self.logger.error(traceback.format_exc())
        except Exception as e:
            self.logger.error(
                f"Error in collect data saving loop for env {env_id}: {e}"
            )
            import traceback

            self.logger.error(traceback.format_exc())

        try:
            self.logger.info(
                f"Saving env info for env {env_id}, obs buffer length: {len(self.record_obs_buffer[env_id])}"
            )
            self._save_env_info(env_id, backend, trajectory_dir)
        except Exception as e:
            self.logger.error(f"Error saving env info for env {env_id}: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

        try:
            if hasattr(backend, "flush"):
                backend.flush()
            elif hasattr(backend, "wait"):
                backend.wait()
        except Exception as e:
            self.logger.warning(f"Backend flush/wait failed (may not be needed): {e}")

        self.logger.info(
            f"Saved trajectory {trajectory_id} for env {env_id} to {trajectory_dir}"
        )

        remaining = self.max_trajectories - self.trajectory_id
        if remaining <= 10 and remaining > 0:
            self.logger.info(
                f"Approaching trajectory limit: {remaining} trajectories remaining (max: {self.max_trajectories})"
            )

    def _save_env_info(
        self,
        env_id: int,
        backend: BackendDispatch,
        trajectory_dir: str,
    ):
        """Save per-step env info for a given env.

        Folder structure under trajectory_dir:
            env/
                camera/
                    cam_0/
                        <annotator_type>/step_0000.*
                        ...
                    cam_1/
                        ...
                info/
                    env_<env_id>_step_0000.json
                    env_<env_id>_step_0001.json
                    ...
        """
        env_infos = self.record_obs_buffer[env_id]

        env_base = os.path.join(trajectory_dir, "env")
        os.makedirs(os.path.join(env_base, "info"), exist_ok=True)
        os.makedirs(os.path.join(env_base, "camera"), exist_ok=True)

        for step_idx, env_info in enumerate(env_infos):
            if step_idx == len(env_infos) - 1:
                continue
            try:
                obs = env_info.get("obs", {})
                policy_obs = obs.get("policy_obs", {})
                privilege_obs = obs.get("privilege_obs", {})

                camera_info = policy_obs.get("camera_info", {})
            except Exception as e:
                self.logger.error(
                    f"Error extracting env_info at step {step_idx} for env {env_id}: {e}"
                )
                import traceback

                self.logger.error(traceback.format_exc())
                continue

            camera_paths = self._save_camera_info_for_step(
                camera_info=camera_info,
                step_idx=step_idx,
                backend=backend,
                trajectory_dir=trajectory_dir,
                env_id=env_id,
            )

            policy_obs_serializable = {}
            for key, value in policy_obs.items():
                if key == "camera_info":
                    policy_obs_serializable["camera_info"] = camera_paths
                else:
                    policy_obs_serializable[key] = self._to_serializable(value)

            obs_serializable = {
                "policy_obs": policy_obs_serializable,
                "privilege_obs": self._to_serializable(privilege_obs),
            }

            reward_serializable = self._to_serializable(env_info.get("reward", None))
            terminated_serializable = self._to_serializable(
                env_info.get("terminated", None)
            )
            truncated_serializable = self._to_serializable(
                env_info.get("truncated", None)
            )
            info_serializable = self._to_serializable(env_info.get("info", {}))

            env_info_serializable = {
                "obs": obs_serializable,
                "reward": reward_serializable,
                "terminated": terminated_serializable,
                "truncated": truncated_serializable,
                "info": info_serializable,
            }

            rel_path = os.path.join(
                "env", "info", f"env_{env_id}_step_{step_idx:04d}.json"
            )
            backend.schedule(
                F.write_json,
                data=env_info_serializable,
                path=rel_path,
                indent=JSON_INDENT,
            )

    def _save_camera_info_for_step(
        self,
        camera_info: Dict[str, Any],
        step_idx: int,
        backend: BackendDispatch,
        trajectory_dir: str,
        env_id: int,
    ) -> Dict[str, Any]:
        """Save camera_info for a single step and return same-structured paths.

        Input format (per env, from _convert_batched_env_info_to_per_env):
            camera_info: {annotator_name: [per_cam_data_0, per_cam_data_1, ...]}

        Output format:
            {annotator_name: [relative_folder_cam0, relative_folder_cam1, ...]}
            where each folder is: camera/cam_{id}/{annotator_name}/
        """
        if not camera_info:
            return {}

        first_value = next(iter(camera_info.values()))
        if not isinstance(first_value, list):
            return self._to_serializable(camera_info)

        num_cams = len(first_value)

        camera_paths: Dict[str, Any] = {
            annotator_name: [None] * num_cams for annotator_name in camera_info.keys()
        }

        for cam_id in range(num_cams):
            cam_base_rel = os.path.join("env", "camera", f"cam_{cam_id}")
            cam_base_abs = os.path.join(trajectory_dir, cam_base_rel)
            os.makedirs(cam_base_abs, exist_ok=True)

            for annotator_name, per_cam_list in camera_info.items():
                if cam_id >= len(per_cam_list):
                    continue

                anno_data = per_cam_list[cam_id]

                annotator_dir_rel = os.path.join(cam_base_rel, annotator_name)
                annotator_dir_abs = os.path.join(trajectory_dir, annotator_dir_rel)
                os.makedirs(annotator_dir_abs, exist_ok=True)

                payload = self._convert_annotator_data_to_payload(
                    annotator_name, anno_data
                )

                if annotator_name == "camera_params":
                    payload = self._add_local_pose_to_camera_params(
                        payload, env_id, cam_id
                    )

                write_annotator_step(
                    backend=backend,
                    annotator_name=annotator_name,
                    payload=payload,
                    annotator_dir_rel=annotator_dir_rel,
                    step_idx=step_idx,
                )

                camera_paths[annotator_name][cam_id] = annotator_dir_rel

        return camera_paths

    def _convert_annotator_data_to_payload(
        self, annotator_name: str, anno_data: Any
    ) -> Dict[str, Any]:
        """Convert annotator.get_data() output to payload format matching TiledCaptureManager.

        This handles the case where anno_data may have {"data": ..., "info": {...}} structure
        and needs to be flattened based on annotator type.
        """

        if not isinstance(anno_data, dict):
            return {"data": anno_data}

        if "info" in anno_data and "data" in anno_data:
            info = anno_data["info"]
            data = anno_data["data"]

            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            elif hasattr(data, "numpy"):
                data = data.numpy()

            if annotator_name == "rgb":
                return {"data": data}

            elif annotator_name == "normals":
                return {"data": data}

            elif annotator_name.startswith("semantic_segmentation"):
                return {
                    "data": data,
                    "idToLabels": info.get("idToLabels", {}),
                }

            elif annotator_name.startswith("instance_id_segmentation"):
                return {
                    "data": data,
                    "idToLabels": info.get("idToLabels", {}),
                }

            elif annotator_name.startswith("instance_segmentation"):
                return {
                    "data": data,
                    "idToLabels": info.get("idToLabels", {}),
                    "idToSemantics": info.get("idToSemantics", {}),
                }

            elif annotator_name.startswith("bounding_box"):
                return {
                    "data": data,
                    "idToLabels": info.get("idToLabels", {}),
                    "primPaths": info.get("primPaths", {}),
                }

            elif annotator_name == "camera_params":
                return anno_data

            elif annotator_name == "pointcloud" or annotator_name == "skeleton_data":
                return anno_data

            else:
                return {"data": data}

        else:
            if "data" in anno_data:
                data = anno_data["data"]
                if hasattr(data, "cpu"):
                    anno_data["data"] = data.cpu().numpy()
                elif hasattr(data, "numpy"):
                    anno_data["data"] = data.numpy()
            return anno_data

    def _add_local_pose_to_camera_params_at_record_time(
        self, camera_params_list: List[Any], env_id: int
    ) -> List[Any]:
        """Add local pose to camera_params at recording time (for each camera).

        This ensures each step records its own pose, not the final pose when saving.
        camera_params_list format: [cam_0_data, cam_1_data, ...]
        """
        try:
            camera_manager = getattr(self.env, "camera_manager", None) or getattr(
                self.env.scene, "camera_manager", None
            )
            if camera_manager is None:
                return camera_params_list

            result_list = []
            for cam_id, cam_params in enumerate(camera_params_list):
                if cam_params is None:
                    result_list.append(None)
                    continue

                if isinstance(cam_params, dict):
                    payload = cam_params.copy()
                else:
                    payload = {"data": cam_params}

                camera_xform = camera_manager.cameras_xform[env_id][cam_id]
                local_pos, local_quat = camera_xform.get_local_pose()

                payload["pos_local"] = self._to_serializable(local_pos)
                payload["ori_local"] = self._to_serializable(local_quat)

                result_list.append(payload)

            return result_list
        except Exception:
            return camera_params_list

    def _add_local_pose_to_camera_params(
        self, payload: Dict[str, Any], env_id: int, cam_id: int
    ) -> Dict[str, Any]:
        """Add local pose (pos and ori relative to parent prim) to camera_params payload.

        Note: This is called during save time. If pos_local and ori_local already exist
        (from recording time), they will be preserved. Otherwise, current pose is used.
        """

        if "pos_local" in payload and "ori_local" in payload:
            return payload

        try:
            camera_manager = getattr(self.env, "camera_manager", None) or getattr(
                self.env.scene, "camera_manager", None
            )
            if camera_manager is None:
                return payload

            camera_xform = camera_manager.cameras_xform[env_id][cam_id]
            local_pos, local_quat = camera_xform.get_local_pose()

            payload["pos_local"] = self._to_serializable(local_pos)
            payload["ori_local"] = self._to_serializable(local_quat)
        except Exception:
            pass

        return payload

    def get_record_buffer(self):
        return self.record_obs_buffer, self.record_action_buffer

    def has_reached_limit(self) -> bool:
        """Check if the trajectory limit has been reached.

        Returns:
            True if the maximum number of trajectories has been reached or exceeded.
        """
        return self._reached_trajectory_limit

    def should_stop_collection(self) -> bool:
        """Check if collection should stop.

        Returns:
            True if trajectory limit has been reached and no more data should be collected.
        """
        return self._reached_trajectory_limit

    def _check_dict_values_length(
        self, data: Dict[str, Any], expected_length: int, path: str = ""
    ) -> None:
        """Recursively check that all innermost values in a dictionary have the expected length.

        Args:
            data: Dictionary to check (can be nested)
            expected_length: Expected length (num_envs)
            path: Current path in the dictionary (for error messages)
        """
        if not isinstance(data, dict):
            return

        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, torch.Tensor):
                if value.ndim > 0:
                    actual_length = value.shape[0]
                    assert actual_length == expected_length, (
                        f"Value at path '{current_path}' has length {actual_length}, "
                        f"expected {expected_length}. Shape: {value.shape}"
                    )

                    assert not torch.isnan(value).any(), (
                        f"Found nan values in tensor at path '{current_path}'. "
                        f"This indicates a bug in padding or data processing."
                    )

            elif isinstance(value, dict):
                self._check_dict_values_length(value, expected_length, current_path)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        if item.ndim > 0:
                            actual_item_length = item.shape[0]
                            assert actual_item_length == expected_length, (
                                f"Value at path '{current_path}[{i}]' has length {actual_item_length}, "
                                f"expected {expected_length}. Shape: {item.shape}"
                            )

                            assert not torch.isnan(item).any(), (
                                f"Found nan values in tensor at path '{current_path}[{i}]'. "
                                f"This indicates a bug in padding or data processing."
                            )
                    elif isinstance(item, dict):
                        self._check_dict_values_length(
                            item, expected_length, f"{current_path}[{i}]"
                        )
