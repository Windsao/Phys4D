import os


def _patch_packaging_compatibility():
    """packaging  _TrimmedRelease  Isaac Sim  setuptools"""
    try:
        from packaging.version import Version, _TrimmedRelease

        return
    except ImportError:
        try:
            import packaging.version as version_module
            from packaging.version import Version

            class _TrimmedRelease(Version):
                """
                Isaac Sim  setuptools
                packaging  Isaac Sim
                """

                @property
                def release(self):
                    """Release segment without any trailing zeros."""
                    rel = super(_TrimmedRelease, self).release

                    last_nonzero = -1
                    for i in range(len(rel) - 1, -1, -1):
                        if rel[i] != 0:
                            last_nonzero = i
                            break

                    if last_nonzero == -1:
                        return (0,)
                    return rel[: last_nonzero + 1]

            version_module._TrimmedRelease = _TrimmedRelease

            if hasattr(version_module, "__all__"):
                if "_TrimmedRelease" not in version_module.__all__:
                    version_module.__all__ = list(version_module.__all__) + [
                        "_TrimmedRelease"
                    ]
        except Exception:
            pass


_patch_packaging_compatibility()


MAGICPHYSICS_HOME = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
MAGICPHYSICS_ASSETS = os.path.join(MAGICPHYSICS_HOME, "Assets")
MAGICPHYSICS_CONF = os.path.join(MAGICPHYSICS_HOME, "src/phys4d/Conf")


try:
    import isaaclab
except ImportError:
    raise RuntimeError(
        "IsaacLab not found. Please follow the instructions in the README to set up the environment."
    )
ISAACLAB_HOME = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(isaaclab.__file__)))
    )
)


try:
    import phys4d.Env  # noqa: F401
except ImportError:
    pass


try:
    from phys4d.Env import (  # noqa: F401
        BallAndBlockEnv,
        BallCollideEnv,
        BallHitsDuckEnv,
        BallHitsNothingEnv,
        BallInBasketEnv,
        BallRampEnv,
        BallRollsOffEnv,
        BallRollsOnGlassEnv,
        BallTrainEnv,
        BlockDominoEnv,
        DominoInJuiceEnv,
        DominoWithSpaceEnv,
        DuckAndDominoEnv,
        DuckFallsInBoxEnv,
        DuckStaticEnv,
        LightOnBlockEnv,
        LightOnMugBlockEnv,
        LightOnMugEnv,
        LightOnSculptureEnv,
        MarbleRunXEnv,
        MarbleRunYEnv,
        MirrorBallFallEnv,
        MirrorBallRotateEnv,
        MirrorTeapotRotateEnv,
        PotatoInWaterEnv,
        RollBehindBoxEnv,
        RollFrontBoxEnv,
        RollInBoxEnv,
        RollingReflectionEnv,
        SilkCoverEnv,
        SmileyBallRotatesEnv,
        StableBlocksEnv,
        TeapotRotatesEnv,
        TwoBallsPassEnv,
        UnstableBlockStackEnv,
        WeightOnCeramicEnv,
        WeightOnPillowEnv,
        WeightProtectsDuckEnv,
        RotorGsEnv,
        RotorGsMoveCameraEnv,
    )
except ImportError:
    pass
