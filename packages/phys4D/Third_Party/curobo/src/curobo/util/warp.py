











import os


import warp as wp
from packaging import version


from curobo.types.base import TensorDeviceType
from curobo.util.logger import log_info


def init_warp(quiet=True, tensor_args: TensorDeviceType = TensorDeviceType()):
    wp.config.quiet = quiet







    wp.init()


    return True


def warp_support_sdf_struct(wp_module=None):
    if wp_module is None:
        wp_module = wp
    wp_version = wp_module.config.version

    if version.parse(wp_version) < version.parse("1.0.0"):
        log_info(
            "Warp version is "
            + wp_version
            + " < 1.0.0, using older sdf kernels."
            + "No issues expected."
        )
        return False
    return True


def warp_support_kernel_key(wp_module=None):
    if wp_module is None:
        wp_module = wp
    wp_version = wp_module.config.version

    if version.parse(wp_version) < version.parse("1.2.1"):
        log_info(
            "Warp version is "
            + wp_version
            + " < 1.2.1, using, creating global constant to trigger kernel generation."
        )
        return False
    return True


def warp_support_bvh_constructor_type(wp_module=None):
    if wp_module is None:
        wp_module = wp
    wp_version = wp_module.config.version

    if version.parse(wp_version) < version.parse("1.6.0"):
        log_info(
            "Warp version is "
            + wp_version
            + " < 1.6.0, using, creating global constant to trigger kernel generation."
        )
        return False
    return True


def is_runtime_warp_kernel_enabled() -> bool:
    env_variable = os.environ.get("CUROBO_WARP_RUNTIME_KERNEL_DISABLE")
    if env_variable is None:
        return True
    return bool(int(env_variable))
