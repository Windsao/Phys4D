




"""Sub-package for marker utilities to simplify creation of UI elements in the GUI.

Currently, the sub-package provides the following classes:

* :class:`VisualizationMarkers` for creating a group of markers using `UsdGeom.PointInstancer
  <https://graphics.pixar.com/usd/dev/api/class_usd_geom_point_instancer.html>`_.


.. note::

    For some simple use-cases, it may be sufficient to use the debug drawing utilities from Isaac Sim.
    The debug drawing API is available in the `isaacsim.util.debug_drawing`_ module. It allows drawing of
    points and splines efficiently on the UI.

    .. _isaacsim.util.debug_drawing: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_debug_drawing.html

"""

from .config import *
from .visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
