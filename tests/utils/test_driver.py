"""Test functions for the ``bonafide.utils.driver`` module."""

import logging
import os
import re
import sys
from pathlib import Path
from subprocess import PIPE, CompletedProcess, run
from typing import Any, Callable, Dict
from unittest.mock import patch

import pytest

from bonafide.utils.driver import external_driver, kallisto_driver, multiwfn_driver, xtb_driver

########################################
# Tests for the xtb_driver() function. #
########################################

XTB_ENV_VARS = {
    "OMP_STACKSIZE": "3g",
    "OMP_NUM_THREADS": 1,
    "OMP_MAX_ACTIVE_LEVELS": 1,
    "MKL_NUM_THREADS": 1,
}


@pytest.mark.xtb_driver
def test_xtb_driver(
    fetch_data_file: Callable[[str], str], check_dependency: Callable[[str], None]
) -> None:
    """Test for the ``xtb_driver()`` function: valid input."""
    check_dependency("xtb")

    # Setup input
    _input_path = fetch_data_file("radical_cation-conf_01.xyz")
    _input_dict = {
        "input_file_path": _input_path,
        "output_file_path": "irrelevant_path.out",
        "iterations": 300,
        "acc": 350,
        "etemp": 400,
        "chrg": 1,
        "uhf": 1,
        "vfukui": None,
    }

    # Run xtb
    with patch("bonafide.utils.driver.run", wraps=run) as spy:
        return_code, stderr = xtb_driver(input_dict=_input_dict, environment_variables=XTB_ENV_VARS)

    spy.assert_called_once()
    xtb_call = spy.call_args[0][0]
    assert xtb_call[0] == "xtb"
    assert xtb_call[1] == _input_path

    assert type(return_code) == int
    assert type(stderr) == str
    assert return_code == 0
    assert stderr == "normal termination of xtb\n"

    _expected_call = (
        f"program call               : xtb {_input_path} --iterations 300 --acc 350 "
        "--etemp 400 --chrg 1 --uhf 1 --vfukui"
    )
    output_content = Path(_input_dict["output_file_path"]).read_text(encoding="utf-8").strip()
    assert _expected_call in output_content

    # Clean up xtb generated files
    os.remove(_input_dict["output_file_path"])
    os.remove("charges")
    os.remove("wbo")
    os.remove("xtbtopo.mol")
    os.remove("xtbrestart")


@pytest.mark.xtb_driver
def test_xtb_driver2(
    fetch_data_file: Callable[[str], str], check_dependency: Callable[[str], None]
) -> None:
    """Test for the ``xtb_driver()`` function: invalid input (xtb will fail)."""
    check_dependency("xtb")

    # Setup input
    _input_path = fetch_data_file("radical_cation-conf_01.xyz")
    _input_dict = {
        "input_file_path": _input_path,
        "output_file_path": "irrelevant_path.out",
        "iterations": 5,
        "acc": 0.01,
        "etemp": 0,
        "chrg": 1,
        "uhf": 1,
        "vfukui": None,
    }

    # Run xtb
    with patch("bonafide.utils.driver.run", wraps=run) as spy:
        return_code, stderr = xtb_driver(input_dict=_input_dict, environment_variables=XTB_ENV_VARS)

    spy.assert_called_once()
    xtb_call = spy.call_args[0][0]
    assert xtb_call[0] == "xtb"
    assert xtb_call[1] == _input_path

    assert return_code == 1
    assert stderr.startswith("abnormal termination of xtb")

    # Clean up
    os.remove(_input_dict["output_file_path"])
    os.remove(".sccnotconverged")


#############################################
# Tests for the multiwfn_driver() function. #
#############################################

MULTIWFN_ENV_VARS = {"OMP_STACKSIZE": "1G", "NUM_THREADS": 4}


@pytest.mark.multiwfn_driver
def test_multiwfn_driver(
    fetch_data_file: Callable[[str], str], check_dependency: Callable[[str], None]
) -> None:
    """Test for the ``multiwfn_driver()`` function."""
    check_dependency("Multiwfn_noGUI")

    # Setup input
    _input_path = fetch_data_file("clopidogrel-conf_05.fchk")
    _output_name = "irrelevant_name"
    _dummy_command = [15, 0, 8, -10, 12, 3, 0.4, -1, "q"]

    # Run Multiwfn
    with patch("bonafide.utils.driver.run", wraps=run) as spy:
        result = multiwfn_driver(
            cmds=_dummy_command,
            input_file_path=_input_path,
            output_file_name=_output_name,
            environment_variables=MULTIWFN_ENV_VARS,
            namespace="irrelevant",
        )

    spy.assert_called_once()
    multiwfn_call = spy.call_args[0][0]
    assert multiwfn_call[0] == "Multiwfn_noGUI"
    assert multiwfn_call[1] == "_bonafide_mwfn_inp.fchk"
    assert len(multiwfn_call) == 2

    assert result is None
    assert os.path.isfile(f"{_output_name}.out")
    assert not os.path.lexists("_bonafide_mwfn_inp.fchk")

    # Check output (ignoring ANSI color codes, which newer Multiwfn versions print)
    with open(f"{_output_name}.out", "r") as f:
        output_content = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in f.readlines()]

    assert (
        output_content[0]
        == r"# Multiwfn commands: 15 \n 0 \n 8 \n -10 \n 12 \n 3 \n 0.4 \n -1 \n q \n" + "\n"
    )
    assert output_content[1] == f"# Multiwfn input file path: {os.path.abspath(_input_path)}\n"
    assert output_content[3] == " Multiwfn -- A Multifunctional Wavefunction Analyzer\n"
    assert output_content[-1] == " 300 Other functions (Part 3)\n"

    _found = False
    for line in output_content:
        if " Loaded _bonafide_mwfn_inp.fchk successfully!\n" == line:
            _found = True
            break
    assert _found is True

    # Clean up
    os.remove(f"{_output_name}.out")


@pytest.mark.multiwfn_driver
def test_multiwfn_driver2(
    fetch_data_file: Callable[[str], str],
    check_dependency: Callable[[str], None],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for the ``multiwfn_driver()`` function with an input file path that is too long to be
    handled by Multiwfn directly (> 200 characters) and a long file name.
    """
    check_dependency("Multiwfn_noGUI")

    # Setup input with a long path
    _long_dir = tmp_path.joinpath("d" * 100, "d" * 100, "d" * 100)
    _long_dir.mkdir(parents=True)
    _input_path = str(_long_dir / ("f" * 200 + ".fchk"))
    with open(fetch_data_file("clopidogrel-conf_05.fchk"), "rb") as src:
        with open(_input_path, "wb") as dst:
            dst.write(src.read())
    assert len(_input_path) > 500

    _work_dir = tmp_path / "work"
    _work_dir.mkdir()
    monkeypatch.chdir(_work_dir)

    # Run Multiwfn
    multiwfn_driver(
        cmds=["q"],
        input_file_path=_input_path,
        output_file_name="irrelevant_name",
        environment_variables=MULTIWFN_ENV_VARS,
        namespace="irrelevant",
    )

    # Only the output file must remain in the working directory
    assert os.listdir(_work_dir) == ["irrelevant_name.out"]
    assert os.path.isfile(_input_path)

    with open("irrelevant_name.out", "r") as f:
        output_content = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in f.readlines()]
    assert output_content[1] == f"# Multiwfn input file path: {_input_path}\n"
    assert " Loaded _bonafide_mwfn_inp.fchk successfully!\n" in output_content


@pytest.mark.multiwfn_driver
def test_multiwfn_driver3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test for the ``multiwfn_driver()`` function if the symbolic link to the input file cannot
    be created and the file is copied instead.
    """
    _input_path = tmp_path / "input.molden"
    _input_path.write_text("dummy content")
    _work_dir = tmp_path / "work"
    _work_dir.mkdir()
    monkeypatch.chdir(_work_dir)

    def _fail_symlink(*args: Any, **kwargs: Any) -> None:
        raise OSError("symlink not possible")

    def _fake_run(args: Any, **kwargs: Any) -> CompletedProcess:
        # The alias must be a regular file copy of the input file
        assert args[1] == "_bonafide_mwfn_inp.molden"
        assert not os.path.islink(args[1])
        with open(args[1], "r") as f:
            assert f.read() == "dummy content"
        return CompletedProcess(args=args, returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("bonafide.utils.driver.os.symlink", _fail_symlink)
    monkeypatch.setattr("bonafide.utils.driver.run", _fake_run)
    monkeypatch.setattr(
        "bonafide.utils.driver.check_dependency_path",
        lambda prg_name: str(tmp_path / "Multiwfn_noGUI"),
    )

    multiwfn_driver(
        cmds=["q"],
        input_file_path=str(_input_path),
        output_file_name="irrelevant_name",
        environment_variables={},
        namespace="irrelevant",
    )

    assert os.listdir(_work_dir) == ["irrelevant_name.out"]
    assert _input_path.read_text() == "dummy content"
    assert "The file is copied to the working directory instead." in caplog.text


@pytest.mark.multiwfn_driver
def test_multiwfn_driver4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that the input file alias of the ``multiwfn_driver()`` function is removed if running
    Multiwfn raises an exception.
    """
    _input_path = tmp_path / "input.fchk"
    _input_path.write_text("dummy content")
    _work_dir = tmp_path / "work"
    _work_dir.mkdir()
    monkeypatch.chdir(_work_dir)

    def _failing_run(args: Any, **kwargs: Any) -> None:
        assert os.path.islink(args[1])
        raise RuntimeError("Multiwfn crashed")

    monkeypatch.setattr("bonafide.utils.driver.run", _failing_run)
    monkeypatch.setattr(
        "bonafide.utils.driver.check_dependency_path",
        lambda prg_name: str(tmp_path / "Multiwfn_noGUI"),
    )

    with pytest.raises(RuntimeError, match="Multiwfn crashed"):
        multiwfn_driver(
            cmds=["q"],
            input_file_path=str(_input_path),
            output_file_name="irrelevant_name",
            environment_variables={},
            namespace="irrelevant",
        )

    assert os.listdir(_work_dir) == []
    assert _input_path.read_text() == "dummy content"


@pytest.mark.multiwfn_driver
@pytest.mark.parametrize("stale_alias_is_link", [True, False])
def test_multiwfn_driver5(
    stale_alias_is_link: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that the ``multiwfn_driver()`` function fails without modifying an already existing
    input file alias (e.g., left over from an interrupted run).
    """
    _input_path = tmp_path / "input.fchk"
    _input_path.write_text("current molecule")
    _other_input_path = tmp_path / "other.fchk"
    _other_input_path.write_text("other molecule")
    _work_dir = tmp_path / "work"
    _work_dir.mkdir()
    monkeypatch.chdir(_work_dir)

    _alias = _work_dir / "_bonafide_mwfn_inp.fchk"
    if stale_alias_is_link:
        os.symlink(_other_input_path, _alias)
    else:
        _alias.write_text("stale copy")

    def _unexpected_run(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Multiwfn must not be run.")

    monkeypatch.setattr("bonafide.utils.driver.run", _unexpected_run)
    monkeypatch.setattr(
        "bonafide.utils.driver.check_dependency_path",
        lambda prg_name: str(tmp_path / "Multiwfn_noGUI"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        multiwfn_driver(
            cmds=["q"],
            input_file_path=str(_input_path),
            output_file_name="irrelevant_name",
            environment_variables={},
            namespace="irrelevant",
        )

    # The existing alias and all input files must remain untouched
    assert os.listdir(_work_dir) == ["_bonafide_mwfn_inp.fchk"]
    assert os.path.islink(_alias) is stale_alias_is_link
    if stale_alias_is_link:
        assert os.readlink(_alias) == str(_other_input_path)
    else:
        assert _alias.read_text() == "stale copy"
    assert _input_path.read_text() == "current molecule"
    assert _other_input_path.read_text() == "other molecule"
    assert "The file is copied to the working directory instead." not in caplog.text


#############################################
# Tests for the kallisto_driver() function. #
#############################################


@pytest.mark.kallisto_driver
def test_kallisto_driver(
    fetch_data_file: Callable[[str], str], check_dependency: Callable[[str], None]
) -> None:
    """Test for the ``kallisto_driver()`` function: valid input"""
    check_dependency("kallisto")

    # Setup input
    _input_path = fetch_data_file("fluoromethanol.xyz")
    _output_name = "irrelevant_name"

    # Run kallisto
    with patch("bonafide.utils.driver.run", wraps=run) as spy:
        stdout, stderr = kallisto_driver(
            input_section=["eeq"], input_file_path=_input_path, output_file_name=_output_name
        )

    spy.assert_called_once()
    kallisto_call = spy.call_args[0][0]
    assert kallisto_call[0] == "kallisto"
    assert kallisto_call[1] == "eeq"

    assert type(stdout) == str
    assert type(stderr) == str
    assert stdout == ""
    assert stderr == ""
    assert os.path.isfile(f"{_output_name}.out")

    # Check output file
    with open(f"{_output_name}.out", "r") as f:
        output_content = f.readlines()
    assert (
        output_content[0]
        == f"# kallisto program call: kallisto eeq --out irrelevant_name.out {_input_path}\n"
    )

    for line in output_content[2:]:
        try:
            float(line)
        except ValueError:
            pytest.fail("Output file contains non-numeric data.")

    # Clean up
    os.remove(f"{_output_name}.out")


@pytest.mark.kallisto_driver
def test_kallisto_driver2(
    fetch_data_file: Callable[[str], str], check_dependency: Callable[[str], None]
) -> None:
    """Test for the ``kallisto_driver()`` function: invalid input"""
    check_dependency("kallisto")

    # Setup input
    _input_path = fetch_data_file("fluoromethanol.fchk")
    _output_name = "irrelevant_name"

    # Run kallisto
    stdout, stderr = kallisto_driver(
        input_section=["eeq"], input_file_path=_input_path, output_file_name=_output_name
    )

    assert type(stdout) == str
    assert type(stderr) == str
    assert stdout == ""
    assert "RuntimeError: Input format erroneous or not implemented." in stderr
    assert os.path.isfile(f"{_output_name}.out") is False


#############################################
# Tests for the external_driver() function. #
#############################################


@pytest.mark.external_driver
def test_external_driver(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``external_driver()`` function: valid input for running python."""
    # Input script for external python execution
    program_input = """import sys
import os
import numpy as np

arr = np.array([1, 2, 3])
if arr.sum() == 6:
    print("Success")
else:
    print("Failure")

with open("script_out_check.out", "w") as f:
    f.write("check")

sys.exit(0)
"""

    result = external_driver(
        program_path=sys.executable,
        program_input=program_input,
        input_file_extension=".py",
        namespace="irrelevant",
        dependencies=["numpy"],
        stdout=PIPE,
        stderr=PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"Success\n"
    assert os.path.isfile("script_out_check.out") is True
    with open("script_out_check.out", "r") as f:
        content = f.read()
    assert content == "check"

    # Check logs
    assert len(caplog.records) == 0

    # Clean up
    os.remove("script_out_check.out")


@pytest.mark.external_driver
def test_external_driver2(caplog: pytest.LogCaptureFixture) -> None:
    """Test for the ``external_driver()`` function: valid non-Python script."""
    # Setup a simple shell script
    program_input = """#!/bin/bash
echo "This is standard output"
echo "This is standard error" >&2
exit 0
"""
    result = external_driver(
        program_path="/bin/bash",
        program_input=program_input,
        input_file_extension=".sh",
        namespace="irrelevant",
        dependencies=["numpy"],  # is ignored for non-Python scripts (this is checked here)
        stdout=PIPE,
        stderr=PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == b"This is standard output\n"
    assert result.stderr == b"This is standard error\n"

    # Check logs
    assert len(caplog.records) == 0


@pytest.mark.external_driver
@pytest.mark.parametrize(
    "program_path, program_input, input_file_extension, namespace, dependencies, "
    "run_kwargs, _expected_error_str, _expected_error_type",
    [
        (
            False,
            "dummy_input",
            ".py",
            "irrelevant",
            [],
            {},
            "Invalid input to 'program_path': must be of type str but obtained",
            TypeError,
        ),
        (
            "dummy_program_name",
            False,
            ".py",
            "irrelevant",
            [],
            {},
            "Invalid input to 'program_input': must be of type str but obtained",
            TypeError,
        ),
        (
            "dummy_program_name",
            "dummy_input",
            None,
            "irrelevant",
            [],
            {},
            "Invalid input to 'input_file_extension': must be of type str but obtained",
            TypeError,
        ),
        (
            "dummy_program_name",
            "dummy_input",
            ".py",
            ["irrelevant"],
            [],
            {},
            "Invalid input to 'namespace': must be of type str but obtained",
            TypeError,
        ),
        (
            "dummy_program_name",
            "dummy_input",
            ".py",
            "irrelevant",
            {"what_is_needed": True},
            {},
            "Invalid input to 'dependencies': must be of type list but obtained",
            TypeError,
        ),
        (
            "dummy_program_name",
            "dummy_input",
            ".py",
            "irrelevant",
            ["numpy", 123],
            {},
            "Invalid input to 'dependencies': all list entries must be of type str but obtained",
            TypeError,
        ),
        (
            sys.executable,
            "dummy_input",
            ".py",
            "irrelevant",
            ["numpy"],
            {"run_will_not_like_me": True},
            "An unexpected error occurred while running the external program at ",
            RuntimeError,
        ),
    ],
)
def test_external_driver3(
    caplog: pytest.LogCaptureFixture,
    program_path: Any,
    program_input: Any,
    input_file_extension: Any,
    namespace: Any,
    dependencies: Any,
    run_kwargs: Dict[str, Any],
    _expected_error_str: str,
    _expected_error_type: Any,
) -> None:
    """Test for the ``external_driver()`` function: invalid input."""
    with pytest.raises(_expected_error_type, match=_expected_error_str):
        external_driver(
            program_path=program_path,
            program_input=program_input,
            input_file_extension=input_file_extension,
            namespace=namespace,
            dependencies=dependencies,
            **run_kwargs,
        )

    # Check logs
    assert len(caplog.records) > 0
    assert any(record.levelno == logging.ERROR for record in caplog.records)


@pytest.mark.external_driver
@pytest.mark.parametrize("check", [True, False])
def test_external_driver4(caplog: pytest.LogCaptureFixture, check: bool) -> None:
    """Test for the ``external_driver()`` function: invalid input script."""
    # Input script for external python execution
    program_input = """import sys
import os
import numpy as np

res = 42 / 0
print("Success")

sys.exit(0)
"""
    if check is False:
        result = external_driver(
            program_path=sys.executable,
            program_input=program_input,
            input_file_extension=".py",
            namespace="irrelevant",
            dependencies=["numpy"],
            stdout=PIPE,
            stderr=PIPE,
            check=check,
        )

        assert result.returncode == 1
        assert result.stdout == b""
        assert b"ZeroDivisionError: division by zero" in result.stderr

        # Check logs
        assert len(caplog.records) == 0

    else:
        with pytest.raises(
            RuntimeError,
            match="An unexpected error occurred while running the external program at ",
        ):
            result = external_driver(
                program_path=sys.executable,
                program_input=program_input,
                input_file_extension=".py",
                namespace="irrelevant",
                dependencies=["numpy"],
                stdout=PIPE,
                stderr=PIPE,
                check=check,
            )

        # Check logs
        assert len(caplog.records) > 0
        assert any(record.levelno == logging.ERROR for record in caplog.records)
