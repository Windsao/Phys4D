




"""Helper functions for SpaceMouse."""
























def convert_buffer(b1, b2):
    """Converts raw SpaceMouse readings to commands.

    Args:
        b1: 8-bit byte
        b2: 8-bit byte

    Returns:
        Scaled value from Space-mouse message
    """
    return _scale_to_control(_to_int16(b1, b2))


"""
Private methods.
"""


def _to_int16(y1, y2):
    """Convert two 8 bit bytes to a signed 16 bit integer.

    Args:
        y1: 8-bit byte
        y2: 8-bit byte

    Returns:
        16-bit integer
    """
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def _scale_to_control(x, axis_scale=350.0, min_v=-1.0, max_v=1.0):
    """Normalize raw HID readings to target range.

    Args:
        x: Raw reading from HID
        axis_scale: (Inverted) scaling factor for mapping raw input value
        min_v: Minimum limit after scaling
        max_v: Maximum limit after scaling

    Returns:
        Clipped, scaled input from HID
    """
    x = x / axis_scale
    return min(max(x, min_v), max_v)
