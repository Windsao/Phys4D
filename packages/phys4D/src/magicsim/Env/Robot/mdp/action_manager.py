"""Action manager for processing actions sent to the environment."""

from __future__ import annotations

import inspect
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.kit.app

from isaaclab.assets import AssetBase

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from isaaclab.managers.manager_term_cfg import ActionTermCfg

if TYPE_CHECKING:
    from magicsim.Env.Environment.Isaac.IsaacRLEnv import IsaacRLEnv
from gymnasium import spaces


class ActionTerm(ManagerTermBase):
    """Base class for action terms.

    The action term is responsible for processing the raw actions sent to the environment
    and applying them to the asset managed by the term. The action term is comprised of two
    operations:

    * Processing of actions: This operation is performed once per **environment step** and
      is responsible for pre-processing the raw actions sent to the environment.
    * Applying actions: This operation is performed once per **simulation step** and is
      responsible for applying the processed actions to the asset managed by the term.
    """

    _action_space: spaces.Box

    def __init__(self, cfg: ActionTermCfg, env: IsaacRLEnv):
        """Initialize the action term.

        Args:
            cfg: The configuration object.
            env: The environment instance.
        """

        super().__init__(cfg, env)

        self._asset: AssetBase = self._env.scene[self.cfg.asset_name]

        self._debug_vis_handle = None

        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties.
    """

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the term."""
        raise NotImplementedError

    @property
    @abstractmethod
    def processed_actions(self) -> torch.Tensor:
        """The actions computed by the term after applying any processing."""
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the action term has a debug visualization implemented."""

        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the action term data.
        Args:
            debug_vis: Whether to visualize the action term data.
        Returns:
            Whether the debug visualization was successfully set. False if the action term does
            not support debug visualization.
        """

        if not self.has_debug_vis_implementation:
            return False

        self._set_debug_vis_impl(debug_vis)

        if debug_vis:
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(
                        event
                    )
                )
        else:
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None

        return True

    @abstractmethod
    def process_actions(self, actions: torch.Tensor, env_ids: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function is called once per environment step by the manager.

        Args:
            actions: The actions to process.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_actions(self):
        """Applies the actions to the asset managed by the term.

        Note:
            This is called at every simulation step by the manager.
        """
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.
        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(
            f"Debug visualization is not implemented for {self.__class__.__name__}."
        )

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.
        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(
            f"Debug visualization is not implemented for {self.__class__.__name__}."
        )


class ActionManager(ManagerBase):
    """Manager for processing and applying actions for a given world.

    The action manager handles the interpretation and application of user-defined
    actions on a given world. It is comprised of different action terms that decide
    the dimension of the expected actions.

    The action manager performs operations at two stages:

    * processing of actions: It splits the input actions to each term and performs any
      pre-processing needed. This should be called once at every environment step.
    * apply actions: This operation typically sets the processed actions into the assets in the
      scene (such as robots). It should be called before every simulation step.
    """

    _action_space: spaces.Dict
    """The action space for the action manager."""

    def __init__(self, cfg: object, env: IsaacRLEnv):
        """Initialize the action manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ActionTermCfg]``).
            env: The environment instance.

        Raises:
            ValueError: If the configuration is None.
        """

        if cfg is None:
            raise ValueError(
                "Action manager configuration is None. Please provide a valid configuration."
            )

        super().__init__(cfg, env)

        self._action = torch.zeros(
            (self.num_envs, self.total_action_dim), device=self.device
        )
        self._prev_action = torch.zeros_like(self._action)
        self._prev_action_env_ids = None
        self._env_ids = None

        self.cfg.debug_vis = False
        for term in self._terms.values():
            self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for action manager."""
        msg = f"<ActionManager> contains {len(self._term_names)} active terms.\n"

        table = PrettyTable()
        table.title = f"Active Action Terms (shape: {self.total_action_dim})"
        table.field_names = ["Index", "Name", "Dimension"]

        table.align["Name"] = "l"
        table.align["Dimension"] = "r"

        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.action_dim])

        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def total_action_dim(self) -> int:
        """Total dimension of actions."""
        return sum(self.action_term_dim)

    @property
    def active_terms(self) -> list[str]:
        """Name of active action terms."""
        return self._term_names

    @property
    def action_term_dim(self) -> list[int]:
        """Shape of each action term."""
        return [term.action_dim for term in self._terms.values()]

    @property
    def action(self) -> torch.Tensor:
        """The actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._action

    @property
    def prev_action(self) -> torch.Tensor:
        """The previous actions sent to the environment. Shape is (num_envs, total_action_dim)."""
        return self._prev_action, self._prev_action_env_ids

    @property
    def env_ids(self):
        return self._env_ids

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the command terms have debug visualization implemented."""

        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """

    def get_active_iterable_terms(
        self, env_idx: int
    ) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        idx = 0
        for name, term in self._terms.items():
            term_actions = self._action[env_idx, idx : idx + term.action_dim].cpu()
            terms.append((name, term_actions.tolist()))
            idx += term.action_dim
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the action data.
        Args:
            debug_vis: Whether to visualize the action data.
        Returns:
            Whether the debug visualization was successfully set. False if the action
            does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Resets the action history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            An empty dictionary.
        """

        if env_ids is None:
            env_ids = slice(None)

        self._prev_action[env_ids] = 0.0
        self._action[env_ids] = 0.0

        for term in self._terms.values():
            term.reset(env_ids=env_ids)

        return {}

    def process_action(self, action: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """

        if self.total_action_dim != action.shape[1]:
            raise ValueError(
                f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}."
            )

        self._prev_action_env_ids = self.env_ids
        self._env_ids = env_ids

        self._prev_action[self._prev_action_env_ids] = self._action[
            self._prev_action_env_ids
        ]
        self._action[self.env_ids] = action

        idx = 0
        for term in self._terms.values():
            term_actions = self._action[env_ids, idx : idx + term.action_dim]
            term.process_actions(term_actions, env_ids)
            idx += term.action_dim

    def apply_action(self) -> None:
        """Applies the actions to the environment/simulation.

        Note:
            This should be called at every simulation step.
        """
        for term in self._terms.values():
            term.apply_actions()

    def get_term(self, name: str) -> ActionTerm:
        """Returns the action term with the specified name.

        Args:
            name: The name of the action term.

        Returns:
            The action term with the specified name.
        """
        return self._terms[name]

    def serialize(self) -> dict:
        """Serialize the action manager configuration.

        Returns:
            A dictionary of serialized action term configurations.
        """
        return {term_name: term.serialize() for term_name, term in self._terms.items()}

    @property
    def action_space(self) -> spaces.Dict:
        """Returns the action space for the action manager.

        Returns:
            The action space for the action manager.
        """
        return self._action_space

    """
    Helper functions.
    """

    def _prepare_terms(self):
        self._term_names: list[str] = list()
        self._terms: dict[str, ActionTerm] = dict()

        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()

        self._action_space = spaces.Dict()

        print(" cfg_items: ", cfg_items)
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue

            if not isinstance(term_cfg, ActionTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ActionTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )

            term = term_cfg.class_type(term_cfg, self._env)

            if not isinstance(term, ActionTerm):
                raise TypeError(
                    f"Returned object for the term '{term_name}' is not of type ActionType."
                )

            self._term_names.append(term_name)
            self._terms[term_name] = term
            self._action_space.spaces[term_name] = term.action_space

    def get_raw_action(self, env_ids: Sequence[int] = None):
        """Get raw actions for each action term.

        Args:
            env_ids: The environment ids to get actions for. If None, returns actions for all environments.

        Returns:
            A dictionary mapping term names to their raw actions tensors.
            If env_ids is provided, the tensors will be indexed to only include those environments.
        """
        raw_actions_dict = {}

        for term_name, term in self._terms.items():
            raw_actions = term.raw_actions

            if env_ids is not None:
                raw_actions = raw_actions[env_ids]

            raw_actions_dict[term_name] = raw_actions

        return raw_actions_dict

    def get_processed_action(self, env_ids: Sequence[int] = None):
        """Get processed actions for each action term.

        Args:
            env_ids: The environment ids to get actions for. If None, returns actions for all environments.

        Returns:
            A dictionary mapping term names to their processed actions tensors.
            If env_ids is provided, the tensors will be indexed to only include those environments.
        """
        processed_actions_dict = {}

        for term_name, term in self._terms.items():
            processed_actions = term.processed_actions
            processed_actions_dict[term_name] = processed_actions

        return processed_actions_dict
