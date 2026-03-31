




"""Event manager for orchestrating operations based on different simulation events."""

from __future__ import annotations

import inspect
import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.log

from .manager_base import ManagerBase
from .manager_term_cfg import EventTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class EventManager(ManagerBase):
    """Manager for orchestrating operations based on different simulation events.

    The event manager applies operations to the environment based on different simulation events. For example,
    changing the masses of objects or their friction coefficients during initialization/ reset, or applying random
    pushes to the robot at a fixed interval of steps. The user can specify several modes of events to fine-tune the
    behavior based on when to apply the event.

    The event terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each event term should instantiate the :class:`EventTermCfg` class.

    Event terms can be grouped by their mode. The mode is a user-defined string that specifies when
    the event term should be applied. This provides the user complete control over when event
    terms should be applied.

    For a typical training process, you may want to apply events in the following modes:

    - "prestartup": Event is applied once at the beginning of the training before the simulation starts.
      This is used to randomize USD-level properties of the simulation stage.
    - "startup": Event is applied once at the beginning of the training once simulation is started.
    - "reset": Event is applied at every reset.
    - "interval": Event is applied at pre-specified intervals of time.

    However, you can also define your own modes and use them in the training process as you see fit.
    For this you will need to add the triggering of that mode in the environment implementation as well.

    .. note::

        The triggering of operations corresponding to the mode ``"interval"`` are the only mode that are
        directly handled by the manager itself. The other modes are handled by the environment implementation.

    """

    _env: ManagerBasedEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedEnv):
        """Initialize the event manager.

        Args:
            cfg: A configuration object or dictionary (``dict[str, EventTermCfg]``).
            env: An environment object.
        """

        self._mode_term_names: dict[str, list[str]] = dict()
        self._mode_term_cfgs: dict[str, list[EventTermCfg]] = dict()
        self._mode_class_term_cfgs: dict[str, list[EventTermCfg]] = dict()


        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Returns: A string representation for event manager."""
        msg = f"<EventManager> contains {len(self._mode_term_names)} active terms.\n"


        for mode in self._mode_term_names:

            table = PrettyTable()
            table.title = f"Active Event Terms in Mode: '{mode}'"

            if mode == "interval":
                table.field_names = ["Index", "Name", "Interval time range (s)"]
                table.align["Name"] = "l"
                for index, (name, cfg) in enumerate(zip(self._mode_term_names[mode], self._mode_term_cfgs[mode])):
                    table.add_row([index, name, cfg.interval_range_s])
            else:
                table.field_names = ["Index", "Name"]
                table.align["Name"] = "l"
                for index, name in enumerate(self._mode_term_names[mode]):
                    table.add_row([index, name])

            msg += table.get_string()
            msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> dict[str, list[str]]:
        """Name of active event terms.

        The keys are the modes of event and the values are the names of the event terms.
        """
        return self._mode_term_names

    @property
    def available_modes(self) -> list[str]:
        """Modes of events."""
        return list(self._mode_term_names.keys())

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:

        for mode_cfg in self._mode_class_term_cfgs.values():
            for term_cfg in mode_cfg:
                term_cfg.func.reset(env_ids=env_ids)


        if env_ids is None:
            num_envs = self._env.num_envs
        else:
            num_envs = len(env_ids)



        if "interval" in self._mode_term_cfgs:
            for index, term_cfg in enumerate(self._mode_term_cfgs["interval"]):



                if not term_cfg.is_global_time:
                    lower, upper = term_cfg.interval_range_s
                    sampled_interval = torch.rand(num_envs, device=self.device) * (upper - lower) + lower
                    self._interval_term_time_left[index][env_ids] = sampled_interval


        return {}

    def apply(
        self,
        mode: str,
        env_ids: Sequence[int] | None = None,
        dt: float | None = None,
        global_env_step_count: int | None = None,
    ):
        """Calls each event term in the specified mode.

        This function iterates over all the event terms in the specified mode and calls the function
        corresponding to the term. The function is called with the environment instance and the environment
        indices to apply the event to.

        For the "interval" mode, the function is called when the time interval has passed. This requires
        specifying the time step of the environment.

        For the "reset" mode, the function is called when the mode is "reset" and the total number of environment
        steps that have happened since the last trigger of the function is equal to its configured parameter for
        the number of environment steps between resets.

        Args:
            mode: The mode of event.
            env_ids: The indices of the environments to apply the event to.
                Defaults to None, in which case the event is applied to all environments when applicable.
            dt: The time step of the environment. This is only used for the "interval" mode.
                Defaults to None to simplify the call for other modes.
            global_env_step_count: The total number of environment steps that have happened. This is only used
                for the "reset" mode. Defaults to None to simplify the call for other modes.

        Raises:
            ValueError: If the mode is ``"interval"`` and the time step is not provided.
            ValueError: If the mode is ``"interval"`` and the environment indices are provided. This is an undefined
                behavior as the environment indices are computed based on the time left for each environment.
            ValueError: If the mode is ``"reset"`` and the total number of environment steps that have happened
                is not provided.
        """

        if mode not in self._mode_term_names:
            omni.log.warn(f"Event mode '{mode}' is not defined. Skipping event.")
            return


        if mode == "interval" and dt is None:
            raise ValueError(f"Event mode '{mode}' requires the time-step of the environment.")
        if mode == "interval" and env_ids is not None:
            raise ValueError(
                f"Event mode '{mode}' does not require environment indices. This is an undefined behavior"
                " as the environment indices are computed based on the time left for each environment."
            )

        if mode == "reset" and global_env_step_count is None:
            raise ValueError(f"Event mode '{mode}' requires the total number of environment steps to be provided.")


        for index, term_cfg in enumerate(self._mode_term_cfgs[mode]):
            if mode == "interval":

                time_left = self._interval_term_time_left[index]

                time_left -= dt



                if term_cfg.is_global_time:
                    if time_left < 1e-6:
                        lower, upper = term_cfg.interval_range_s
                        sampled_interval = torch.rand(1) * (upper - lower) + lower
                        self._interval_term_time_left[index][:] = sampled_interval


                        term_cfg.func(self._env, None, **term_cfg.params)
                else:
                    valid_env_ids = (time_left < 1e-6).nonzero().flatten()
                    if len(valid_env_ids) > 0:
                        lower, upper = term_cfg.interval_range_s
                        sampled_time = torch.rand(len(valid_env_ids), device=self.device) * (upper - lower) + lower
                        self._interval_term_time_left[index][valid_env_ids] = sampled_time


                        term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
            elif mode == "reset":

                min_step_count = term_cfg.min_step_count_between_reset

                if env_ids is None:
                    env_ids = slice(None)



                if min_step_count == 0:
                    self._reset_term_last_triggered_step_id[index][env_ids] = global_env_step_count
                    self._reset_term_last_triggered_once[index][env_ids] = True


                    term_cfg.func(self._env, env_ids, **term_cfg.params)
                else:

                    last_triggered_step = self._reset_term_last_triggered_step_id[index][env_ids]
                    triggered_at_least_once = self._reset_term_last_triggered_once[index][env_ids]

                    steps_since_triggered = global_env_step_count - last_triggered_step


                    valid_trigger = steps_since_triggered >= min_step_count


                    valid_trigger |= (last_triggered_step == 0) & ~triggered_at_least_once


                    if env_ids == slice(None):
                        valid_env_ids = valid_trigger.nonzero().flatten()
                    else:
                        valid_env_ids = env_ids[valid_trigger]


                    if len(valid_env_ids) > 0:
                        self._reset_term_last_triggered_once[index][valid_env_ids] = True
                        self._reset_term_last_triggered_step_id[index][valid_env_ids] = global_env_step_count


                        term_cfg.func(self._env, valid_env_ids, **term_cfg.params)
            else:

                term_cfg.func(self._env, env_ids, **term_cfg.params)

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: EventTermCfg):
        """Sets the configuration of the specified term into the manager.

        The method finds the term by name by searching through all the modes.
        It then updates the configuration of the term with the first matching name.

        Args:
            term_name: The name of the event term.
            cfg: The configuration for the event term.

        Raises:
            ValueError: If the term name is not found.
        """
        term_found = False
        for mode, terms in self._mode_term_names.items():
            if term_name in terms:
                self._mode_term_cfgs[mode][terms.index(term_name)] = cfg
                term_found = True
                break
        if not term_found:
            raise ValueError(f"Event term '{term_name}' not found.")

    def get_term_cfg(self, term_name: str) -> EventTermCfg:
        """Gets the configuration for the specified term.

        The method finds the term by name by searching through all the modes.
        It then returns the configuration of the term with the first matching name.

        Args:
            term_name: The name of the event term.

        Returns:
            The configuration of the event term.

        Raises:
            ValueError: If the term name is not found.
        """
        for mode, terms in self._mode_term_names.items():
            if term_name in terms:
                return self._mode_term_cfgs[mode][terms.index(term_name)]
        raise ValueError(f"Event term '{term_name}' not found.")

    """
    Helper functions.
    """

    def _prepare_terms(self):


        self._interval_term_time_left: list[torch.Tensor] = list()

        self._reset_term_last_triggered_step_id: list[torch.Tensor] = list()
        self._reset_term_last_triggered_once: list[torch.Tensor] = list()


        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        for term_name, term_cfg in cfg_items:

            if term_cfg is None:
                continue

            if not isinstance(term_cfg, EventTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type EventTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )

            if term_cfg.mode != "reset" and term_cfg.min_step_count_between_reset != 0:
                omni.log.warn(
                    f"Event term '{term_name}' has 'min_step_count_between_reset' set to a non-zero value"
                    " but the mode is not 'reset'. Ignoring the 'min_step_count_between_reset' value."
                )


            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)


            if term_cfg.mode == "prestartup" and self._env.scene.cfg.replicate_physics:
                raise RuntimeError(
                    "Scene replication is enabled, which may affect USD-level randomization."
                    " When assets are replicated, their properties are shared across instances,"
                    " potentially leading to unintended behavior."
                    " For stable USD-level randomization, please disable scene replication"
                    " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
                )




            if inspect.isclass(term_cfg.func) and term_cfg.mode == "prestartup":
                omni.log.info(f"Initializing term '{term_name}' with class '{term_cfg.func.__name__}'.")
                term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)


            if term_cfg.mode not in self._mode_term_names:

                self._mode_term_names[term_cfg.mode] = list()
                self._mode_term_cfgs[term_cfg.mode] = list()
                self._mode_class_term_cfgs[term_cfg.mode] = list()

            self._mode_term_names[term_cfg.mode].append(term_name)
            self._mode_term_cfgs[term_cfg.mode].append(term_cfg)


            if inspect.isclass(term_cfg.func):
                self._mode_class_term_cfgs[term_cfg.mode].append(term_cfg)



            if term_cfg.mode == "interval":
                if term_cfg.interval_range_s is None:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'interval' but 'interval_range_s' is not specified."
                    )


                if term_cfg.is_global_time:
                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(1) * (upper - lower) + lower
                    self._interval_term_time_left.append(time_left)
                else:

                    lower, upper = term_cfg.interval_range_s
                    time_left = torch.rand(self.num_envs, device=self.device) * (upper - lower) + lower
                    self._interval_term_time_left.append(time_left)

            elif term_cfg.mode == "reset":
                if term_cfg.min_step_count_between_reset < 0:
                    raise ValueError(
                        f"Event term '{term_name}' has mode 'reset' but 'min_step_count_between_reset' is"
                        f" negative: {term_cfg.min_step_count_between_reset}. Please provide a non-negative value."
                    )


                step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
                self._reset_term_last_triggered_step_id.append(step_count)

                no_trigger = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                self._reset_term_last_triggered_once.append(no_trigger)
