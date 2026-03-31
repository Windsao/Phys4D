




from __future__ import annotations

import os

import isaacsim
import omni.kit.commands
import omni.usd

from .asset_converter_base import AssetConverterBase
from .mjcf_converter_cfg import MjcfConverterCfg


class MjcfConverter(AssetConverterBase):
    """Converter for a MJCF description file to a USD file.

    This class wraps around the `isaacsim.asset.importer.mjcf`_ extension to provide a lazy implementation
    for MJCF to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the MJCF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 4.5 onwards, the extension name changed from ``omni.importer.mjcf`` to
        ``isaacsim.asset.importer.mjcf``. This converter class now uses the latest extension from Isaac Sim.

    .. _isaacsim.asset.importer.mjcf:  https://docs.isaacsim.omniverse.nvidia.com/latest/importer_exporter/ext_isaacsim_asset_importer_mjcf.html
    """

    cfg: MjcfConverterCfg
    """The configuration instance for MJCF to USD conversion."""

    def __init__(self, cfg: MjcfConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self, cfg: MjcfConverterCfg):
        """Calls underlying Omniverse command to convert MJCF to USD.

        Args:
            cfg: The configuration instance for MJCF to USD conversion.
        """
        import_config = self._get_mjcf_import_config()
        file_basename, _ = os.path.basename(cfg.asset_path).split(".")
        omni.kit.commands.execute(
            "MJCFCreateAsset",
            mjcf_path=cfg.asset_path,
            import_config=import_config,
            dest_path=self.usd_path,
            prim_path=f"/{file_basename}",
        )

    def _get_mjcf_import_config(self) -> isaacsim.asset.importer.mjcf.ImportConfig:
        """Returns the import configuration for MJCF to USD conversion.

        Returns:
            The constructed ``ImportConfig`` object containing the desired settings.
        """

        _, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")








        import_config.set_import_sites(True)



        import_config.set_make_instanceable(self.cfg.make_instanceable)
        import_config.set_instanceable_usd_path(self.usd_instanceable_meshes_path)



        import_config.set_density(self.cfg.link_density)

        import_config.set_import_inertia_tensor(self.cfg.import_inertia_tensor)



        import_config.set_fix_base(self.cfg.fix_base)

        import_config.set_self_collision(self.cfg.self_collision)

        return import_config
