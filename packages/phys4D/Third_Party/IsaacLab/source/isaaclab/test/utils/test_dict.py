






"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher


simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import random

import isaaclab.utils.dict as dict_utils


def _test_function(x):
    """Test function for string <-> callable conversion."""
    return x**2


def _test_lambda_function(x):
    """Test function for string <-> callable conversion."""
    return x**2


def test_print_dict():
    """Test printing of dictionary."""

    test_dict = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4, "f": {"g": 5, "h": 6}},
        "i": 7,
        "j": lambda x: x**2,
        "k": dict_utils.class_to_dict,
    }

    dict_utils.print_dict(test_dict)


def test_string_callable_function_conversion():
    """Test string <-> callable conversion for function."""


    test_string = dict_utils.callable_to_string(_test_function)

    test_function_2 = dict_utils.string_to_callable(test_string)

    assert _test_function(2) == test_function_2(2)


def test_string_callable_function_with_lambda_in_name_conversion():
    """Test string <-> callable conversion for function which has lambda in its name."""


    test_string = dict_utils.callable_to_string(_test_lambda_function)

    test_function_2 = dict_utils.string_to_callable(test_string)

    assert _test_function(2) == test_function_2(2)


def test_string_callable_lambda_conversion():
    """Test string <-> callable conversion for lambda expression."""


    func = lambda x: x**2

    test_string = dict_utils.callable_to_string(func)

    func_2 = dict_utils.string_to_callable(test_string)

    assert test_string == "lambda x: x**2"
    assert func(2) == func_2(2)


def test_dict_to_md5():
    """Test MD5 hash generation for dictionary."""

    test_dict = {
        "a": 1,
        "b": 2,
        "c": {"d": 3, "e": 4, "f": {"g": 5, "h": 6}},
        "i": random.random(),
        "k": dict_utils.callable_to_string(dict_utils.class_to_dict),
    }

    md5_hash_1 = dict_utils.dict_to_md5_hash(test_dict)


    for _ in range(200):
        md5_hash_2 = dict_utils.dict_to_md5_hash(test_dict)
        assert md5_hash_1 == md5_hash_2
