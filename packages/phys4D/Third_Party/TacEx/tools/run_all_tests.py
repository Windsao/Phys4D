




"""A runner script for all the tests within source directory.

.. code-block:: bash

    ./isaaclab.sh -p tools/run_all_tests.py

    # for dry run
    ./isaaclab.sh -p tools/run_all_tests.py --discover_only

    # for quiet run
    ./isaaclab.sh -p tools/run_all_tests.py --quiet

    # for increasing timeout (default is 600 seconds)
    ./isaaclab.sh -p tools/run_all_tests.py --timeout 1000

"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from prettytable import PrettyTable


from test_settings import DEFAULT_TIMEOUT, ISAACLAB_PATH, PER_TEST_TIMEOUTS, TESTS_TO_SKIP


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run all tests under current directory.")

    parser.add_argument(
        "--skip_tests",
        default="",
        help="Space separated list of tests to skip in addition to those in tests_to_skip.py.",
        type=str,
        nargs="*",
    )


    default_test_dir = os.path.join(ISAACLAB_PATH, "source")

    parser.add_argument(
        "--test_dir", type=str, default=default_test_dir, help="Path to the directory containing the tests."
    )


    log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    default_log_path = os.path.join(ISAACLAB_PATH, "logs", "test_results", log_file_name)

    parser.add_argument(
        "--log_path", type=str, default=default_log_path, help="Path to the log file to store the results in."
    )
    parser.add_argument("--discover_only", action="store_true", help="Only discover and print tests, don't run them.")
    parser.add_argument("--quiet", action="store_true", help="Don't print to console, only log to file.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout for each test in seconds.")
    parser.add_argument("--extension", type=str, default=None, help="Run tests only for the given extension.")

    args = parser.parse_args()
    return args


def test_all(
    test_dir: str,
    tests_to_skip: list[str],
    log_path: str,
    timeout: float = DEFAULT_TIMEOUT,
    per_test_timeouts: dict[str, float] = {},
    discover_only: bool = False,
    quiet: bool = False,
    extension: str | None = None,
) -> bool:
    """Run all tests under the given directory.

    Args:
        test_dir: Path to the directory containing the tests.
        tests_to_skip: List of tests to skip.
        log_path: Path to the log file to store the results in.
        timeout: Timeout for each test in seconds. Defaults to DEFAULT_TIMEOUT.
        per_test_timeouts: A dictionary of tests and their timeouts in seconds. Any tests not listed here will use the
            timeout specified by `timeout`. Defaults to an empty dictionary.
        discover_only: If True, only discover and print the tests without running them. Defaults to False.
        quiet: If False, print the output of the tests to the terminal console (in addition to the log file).
            Defaults to False.
        extension: Run tests only for the given extension. Defaults to None, which means all extensions'
            tests will be run.
    Returns:
        True if all un-skipped tests pass or `discover_only` is True. Otherwise, False.

    Raises:
        ValueError: If any test to skip is not found under the given `test_dir`.

    """

    os.makedirs(os.path.dirname(log_path), exist_ok=True)


    logging_handlers = [logging.FileHandler(log_path)]

    if not quiet:
        logging_handlers.append(logging.StreamHandler())

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=logging_handlers)

    all_test_paths, test_paths, skipped_test_paths, test_timeouts = extract_tests_and_timeouts(
        test_dir, extension, tests_to_skip, timeout, per_test_timeouts
    )


    logging.info("\n" + "=" * 60 + "\n")
    logging.info(f"The following {len(all_test_paths)} tests were found:")
    for i, test_path in enumerate(all_test_paths):
        logging.info(f"{i + 1:02d}: {test_path}, timeout: {test_timeouts[test_path]}")
    logging.info("\n" + "=" * 60 + "\n")

    logging.info(f"The following {len(skipped_test_paths)} tests are marked to be skipped:")
    for i, test_path in enumerate(skipped_test_paths):
        logging.info(f"{i + 1:02d}: {test_path}")
    logging.info("\n" + "=" * 60 + "\n")


    if discover_only:
        return True

    results = {}


    for test_path in test_paths:
        results[test_path] = {}
        before = time.time()
        logging.info("\n" + "-" * 60 + "\n")
        logging.info(f"[INFO] Running '{test_path}'\n")
        try:
            completed_process = subprocess.run(
                [sys.executable, test_path], check=True, capture_output=True, timeout=test_timeouts[test_path]
            )
        except subprocess.TimeoutExpired as e:
            logging.error(f"Timeout occurred: {e}")
            result = "TIMEDOUT"
            stdout = e.stdout
            stderr = e.stderr
        except subprocess.CalledProcessError as e:



            result = "FAILED"
            stdout = e.stdout
            stderr = e.stderr
        except Exception as e:
            logging.error(f"Unexpected exception {e}. Please report this issue on the repository.")
            result = "FAILED"
            stdout = None
            stderr = None
        else:
            result = "COMPLETED"
            stdout = completed_process.stdout
            stderr = completed_process.stderr

        after = time.time()
        time_elapsed = after - before


        stdout = stdout.decode("utf-8") if stdout is not None else ""
        stderr = stderr.decode("utf-8") if stderr is not None else ""

        if result == "COMPLETED":

            success_pattern = r"Ran \d+ tests? in [\d.]+s\s+OK"
            if re.search(success_pattern, stdout) or re.search(success_pattern, stderr):
                result = "PASSED"
            else:
                result = "FAILED"


        logging.info(stdout)
        logging.info(stderr)
        logging.info(f"[INFO] Time elapsed: {time_elapsed:.2f} s")
        logging.info(f"[INFO] Result '{test_path}': {result}")

        results[test_path]["time_elapsed"] = time_elapsed
        results[test_path]["result"] = result


    num_tests = len(all_test_paths)
    num_passing = len([test_path for test_path in test_paths if results[test_path]["result"] == "PASSED"])
    num_failing = len([test_path for test_path in test_paths if results[test_path]["result"] == "FAILED"])
    num_timing_out = len([test_path for test_path in test_paths if results[test_path]["result"] == "TIMEDOUT"])
    num_skipped = len(skipped_test_paths)

    if num_tests == 0:
        passing_percentage = 100
    else:
        passing_percentage = (num_passing + num_skipped) / num_tests * 100


    summary_str = "\n\n"
    summary_str += "===================\n"
    summary_str += "Test Result Summary\n"
    summary_str += "===================\n"

    summary_str += f"Total: {num_tests}\n"
    summary_str += f"Passing: {num_passing}\n"
    summary_str += f"Failing: {num_failing}\n"
    summary_str += f"Skipped: {num_skipped}\n"
    summary_str += f"Timing Out: {num_timing_out}\n"

    summary_str += f"Passing Percentage: {passing_percentage:.2f}%\n"


    total_time = sum([results[test_path]["time_elapsed"] for test_path in test_paths])

    summary_str += f"Total Time Elapsed: {total_time // 3600}h"
    summary_str += f"{total_time // 60 % 60}m"
    summary_str += f"{total_time % 60:.2f}s"

    summary_str += "\n\n=======================\n"
    summary_str += "Per Test Result Summary\n"
    summary_str += "=======================\n"


    per_test_result_table = PrettyTable(field_names=["Test Path", "Result", "Time (s)"])
    per_test_result_table.align["Test Path"] = "l"
    per_test_result_table.align["Time (s)"] = "r"
    for test_path in test_paths:
        per_test_result_table.add_row(
            [test_path, results[test_path]["result"], f"{results[test_path]['time_elapsed']:0.2f}"]
        )

    for test_path in skipped_test_paths:
        per_test_result_table.add_row([test_path, "SKIPPED", "N/A"])

    summary_str += per_test_result_table.get_string()


    logging.info(summary_str)


    return num_failing + num_timing_out == 0


def extract_tests_and_timeouts(
    test_dir: str,
    extension: str | None = None,
    tests_to_skip: list[str] = [],
    timeout: float = DEFAULT_TIMEOUT,
    per_test_timeouts: dict[str, float] = {},
) -> tuple[list[str], list[str], list[str], dict[str, float]]:
    """Extract all tests under the given directory or extension and their respective timeouts.

    Args:
        test_dir: Path to the directory containing the tests.
        extension: Run tests only for the given extension. Defaults to None, which means all extensions'
            tests will be run.
        tests_to_skip: List of tests to skip.
        timeout: Timeout for each test in seconds. Defaults to DEFAULT_TIMEOUT.
        per_test_timeouts: A dictionary of tests and their timeouts in seconds. Any tests not listed here will use the
            timeout specified by `timeout`. Defaults to an empty dictionary.

    Returns:
        A tuple containing the paths of all tests, tests to run, tests to skip, and their respective timeouts.

    Raises:
        ValueError: If any test to skip is not found under the given `test_dir`.
    """


    all_test_paths = [str(path) for path in Path(test_dir).resolve().rglob("*test_*.py")]
    skipped_test_paths = []
    test_paths = []

    for test_to_skip in tests_to_skip:
        for test_path in all_test_paths:
            if test_to_skip in test_path:
                break
        else:
            raise ValueError(f"Test to skip '{test_to_skip}' not found in tests.")


    if extension is not None:
        all_tests_in_selected_extension = []

        for test_path in all_test_paths:

            extension_name = test_path[test_path.find("extensions") :].split("/")[1]


            if extension_name != extension:
                continue

            all_tests_in_selected_extension.append(test_path)

        all_test_paths = all_tests_in_selected_extension


    if len(tests_to_skip) != 0:
        for test_path in all_test_paths:
            if any([test_to_skip in test_path for test_to_skip in tests_to_skip]):
                skipped_test_paths.append(test_path)
            else:
                test_paths.append(test_path)
    else:
        test_paths = all_test_paths


    all_test_paths.sort()
    test_paths.sort()
    skipped_test_paths.sort()


    test_timeouts = {test_path: timeout for test_path in all_test_paths}


    for test_path_with_timeout, test_timeout in per_test_timeouts.items():
        for test_path in all_test_paths:
            if test_path_with_timeout in test_path:
                test_timeouts[test_path] = test_timeout

    return all_test_paths, test_paths, skipped_test_paths, test_timeouts


def warm_start_app():
    """Warm start the app to compile shaders before running the tests."""

    print("[INFO] Warm starting the simulation app before running tests.")
    before = time.time()

    warm_start_output = subprocess.run(
        [
            sys.executable,
            "-c",
            "from isaaclab.app import AppLauncher; app_launcher = AppLauncher(headless=True); app_launcher.app.close()",
        ],
        capture_output=True,
    )
    if len(warm_start_output.stderr) > 0:
        if "omni::fabric::IStageReaderWriter" not in str(warm_start_output.stderr) and "scaling_governor" not in str(
            warm_start_output.stderr
        ):
            logging.error(f"Error warm starting the app: {str(warm_start_output.stderr)}")
            exit(1)


    warm_start_rendering_output = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from isaaclab.app import AppLauncher; app_launcher = AppLauncher(headless=True,"
                " enable_cameras=True); app_launcher.app.close()"
            ),
        ],
        capture_output=True,
    )
    if len(warm_start_rendering_output.stderr) > 0:
        if "omni::fabric::IStageReaderWriter" not in str(
            warm_start_rendering_output.stderr
        ) and "scaling_governor" not in str(warm_start_output.stderr):
            logging.error(f"Error warm starting the app with rendering: {str(warm_start_rendering_output.stderr)}")
            exit(1)

    after = time.time()
    time_elapsed = after - before
    print(f"[INFO] Warm start completed successfully in {time_elapsed:.2f} s")


if __name__ == "__main__":

    args = parse_args()


    warm_start_app()


    tests_to_skip = TESTS_TO_SKIP
    tests_to_skip += args.skip_tests


    test_success = test_all(
        test_dir=args.test_dir,
        tests_to_skip=tests_to_skip,
        log_path=args.log_path,
        timeout=args.timeout,
        per_test_timeouts=PER_TEST_TIMEOUTS,
        discover_only=args.discover_only,
        quiet=args.quiet,
        extension=args.extension,
    )

    if not test_success:
        exit(1)
