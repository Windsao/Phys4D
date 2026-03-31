




"""Sub-module for utilities for working with dictionaries."""

import collections.abc
import hashlib
import json
import torch
from collections.abc import Iterable, Mapping, Sized
from typing import Any

from .array import TENSOR_TYPE_CONVERSIONS, TENSOR_TYPES
from .string import callable_to_string, string_to_callable, string_to_slice

"""
Dictionary <-> Class operations.
"""


def class_to_dict(obj: object) -> dict[str, Any]:
    """Convert an object into dictionary recursively.

    Note:
        Ignores all names starting with "__" (i.e. built-in methods).

    Args:
        obj: An instance of a class to convert.

    Raises:
        ValueError: When input argument is not an object.

    Returns:
        Converted dictionary mapping.
    """

    if not hasattr(obj, "__class__"):
        raise ValueError(f"Expected a class instance. Received: {type(obj)}.")

    if isinstance(obj, dict):
        obj_dict = obj
    elif isinstance(obj, torch.Tensor):



        return obj
    elif hasattr(obj, "__dict__"):
        obj_dict = obj.__dict__
    else:
        return obj


    data = dict()
    for key, value in obj_dict.items():

        if key.startswith("__"):
            continue

        if callable(value):
            data[key] = callable_to_string(value)

        elif hasattr(value, "__dict__") or isinstance(value, dict):
            data[key] = class_to_dict(value)

        elif isinstance(value, (list, tuple)):
            data[key] = type(value)([class_to_dict(v) for v in value])
        else:
            data[key] = value
    return data


def update_class_from_dict(obj, data: dict[str, Any], _ns: str = "") -> None:
    """Reads a dictionary and sets object variables recursively.

    This function performs in-place update of the class member attributes.

    Args:
        obj: An instance of a class to update.
        data: Input dictionary to update from.
        _ns: Namespace of the current object. This is useful for nested configuration
            classes or dictionaries. Defaults to "".

    Raises:
        TypeError: When input is not a dictionary.
        ValueError: When dictionary has a value that does not match default config type.
        KeyError: When dictionary has a key that does not exist in the default config type.
    """
    for key, value in data.items():

        key_ns = _ns + "/" + key


        if hasattr(obj, key) or (isinstance(obj, dict) and key in obj):
            obj_mem = obj[key] if isinstance(obj, dict) else getattr(obj, key)


            if isinstance(value, Mapping):

                update_class_from_dict(obj_mem, value, _ns=key_ns)
                continue


            if isinstance(value, Iterable) and not isinstance(value, str):


                if all(not isinstance(el, Mapping) for el in value):
                    out_val = tuple(value) if isinstance(obj_mem, tuple) else value
                    if isinstance(obj, dict):
                        obj[key] = out_val
                    else:
                        setattr(obj, key, out_val)
                    continue


                if obj_mem is None:
                    raise ValueError(
                        f"[Config]: Cannot merge list under namespace: {key_ns} because the existing value is None."
                    )


                if isinstance(obj_mem, Sized) and isinstance(value, Sized) and len(obj_mem) != len(value):
                    raise ValueError(
                        f"[Config]: Incorrect length under namespace: {key_ns}."
                        f" Expected: {len(obj_mem)}, Received: {len(value)}."
                    )


                if isinstance(obj_mem, tuple):
                    value = tuple(value)
                else:
                    set_obj = True

                    for i in range(len(obj_mem)):
                        if isinstance(value[i], Mapping):
                            update_class_from_dict(obj_mem[i], value[i], _ns=key_ns)
                            set_obj = False

                    if not set_obj:
                        continue


            elif callable(obj_mem):

                value = string_to_callable(value)


            elif value is None or isinstance(value, type(obj_mem)):
                pass


            else:
                raise ValueError(
                    f"[Config]: Incorrect type under namespace: {key_ns}."
                    f" Expected: {type(obj_mem)}, Received: {type(value)}."
                )


            if isinstance(obj, dict):
                obj[key] = value
            else:
                setattr(obj, key, value)


        else:
            raise KeyError(f"[Config]: Key not found under namespace: {key_ns}.")


"""
Dictionary <-> Hashable operations.
"""


def dict_to_md5_hash(data: object) -> str:
    """Convert a dictionary into a hashable key using MD5 hash.

    Args:
        data: Input dictionary or configuration object to convert.

    Returns:
        A string object of double length containing only hexadecimal digits.
    """

    if isinstance(data, dict):
        encoded_buffer = json.dumps(data, sort_keys=True).encode()
    else:
        encoded_buffer = json.dumps(class_to_dict(data), sort_keys=True).encode()

    data_hash = hashlib.md5()
    data_hash.update(encoded_buffer)

    return data_hash.hexdigest()


"""
Dictionary operations.
"""


def convert_dict_to_backend(
    data: dict, backend: str = "numpy", array_types: Iterable[str] = ("numpy", "torch", "warp")
) -> dict:
    """Convert all arrays or tensors in a dictionary to a given backend.

    This function iterates over the dictionary, converts all arrays or tensors with the given types to
    the desired backend, and stores them in a new dictionary. It also works with nested dictionaries.

    Currently supported backends are "numpy", "torch", and "warp".

    Note:
        This function only converts arrays or tensors. Other types of data are left unchanged. Mutable types
        (e.g. lists) are referenced by the new dictionary, so they are not copied.

    Args:
        data: An input dict containing array or tensor data as values.
        backend: The backend ("numpy", "torch", "warp") to which arrays in this dict should be converted.
            Defaults to "numpy".
        array_types: A list containing the types of arrays that should be converted to
            the desired backend. Defaults to ("numpy", "torch", "warp").

    Raises:
        ValueError: If the specified ``backend`` or ``array_types`` are unknown, i.e. not in the list of supported
            backends ("numpy", "torch", "warp").

    Returns:
        The updated dict with the data converted to the desired backend.
    """


    if backend not in TENSOR_TYPE_CONVERSIONS:
        raise ValueError(f"Unknown backend '{backend}'. Supported backends are 'numpy', 'torch', and 'warp'.")

    tensor_type_conversions = TENSOR_TYPE_CONVERSIONS[backend]


    parsed_types = list()
    for t in array_types:

        if t not in TENSOR_TYPES:
            raise ValueError(f"Unknown array type: '{t}'. Supported array types are 'numpy', 'torch', and 'warp'.")

        if t == backend:
            continue

        parsed_types.append(TENSOR_TYPES[t])


    output_dict = dict()
    for key, value in data.items():

        data_type = type(value)

        if data_type in parsed_types:

            if data_type not in tensor_type_conversions:
                raise ValueError(f"No registered conversion for data type: {data_type} to {backend}!")

            output_dict[key] = tensor_type_conversions[data_type](value)

        elif isinstance(data[key], dict):
            output_dict[key] = convert_dict_to_backend(value)

        else:
            output_dict[key] = value

    return output_dict


def update_dict(orig_dict: dict, new_dict: collections.abc.Mapping) -> dict:
    """Updates existing dictionary with values from a new dictionary.

    This function mimics the dict.update() function. However, it works for
    nested dictionaries as well.

    Args:
        orig_dict: The original dictionary to insert items to.
        new_dict: The new dictionary to insert items from.

    Returns:
        The updated dictionary.
    """
    for keyname, value in new_dict.items():
        if isinstance(value, collections.abc.Mapping):
            orig_dict[keyname] = update_dict(orig_dict.get(keyname, {}), value)
        else:
            orig_dict[keyname] = value
    return orig_dict


def replace_slices_with_strings(data: dict) -> dict:
    """Replace slice objects with their string representations in a dictionary.

    Args:
        data: The dictionary to process.

    Returns:
        The dictionary with slice objects replaced by their string representations.
    """
    if isinstance(data, dict):
        return {k: replace_slices_with_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_slices_with_strings(v) for v in data]
    elif isinstance(data, slice):
        return f"slice({data.start},{data.stop},{data.step})"
    else:
        return data


def replace_strings_with_slices(data: dict) -> dict:
    """Replace string representations of slices with slice objects in a dictionary.

    Args:
        data: The dictionary to process.

    Returns:
        The dictionary with string representations of slices replaced by slice objects.
    """
    if isinstance(data, dict):
        return {k: replace_strings_with_slices(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_strings_with_slices(v) for v in data]
    elif isinstance(data, str) and data.startswith("slice("):
        return string_to_slice(data)
    else:
        return data


def print_dict(val, nesting: int = -4, start: bool = True):
    """Outputs a nested dictionary."""
    if isinstance(val, dict):
        if not start:
            print("")
        nesting += 4
        for k in val:
            print(nesting * " ", end="")
            print(k, end=": ")
            print_dict(val[k], nesting, start=False)
    else:

        if callable(val):
            print(callable_to_string(val))
        else:
            print(val)
