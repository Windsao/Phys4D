




"""Utility function for version comparison."""


def compare_versions(v1: str, v2: str) -> int:
    parts1 = list(map(int, v1.split(".")))
    parts2 = list(map(int, v2.split(".")))


    length = max(len(parts1), len(parts2))
    parts1 += [0] * (length - len(parts1))
    parts2 += [0] * (length - len(parts2))

    if parts1 > parts2:
        return 1
    elif parts1 < parts2:
        return -1
    else:
        return 0
