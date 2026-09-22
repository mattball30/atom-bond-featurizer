"""Check the installation of BONAFIDE's dependencies.

Run with ``bonafide-check`` (or ``pixi run check`` in BONAFIDE's pixi workspace).
"""

import importlib
import os
import shutil
import tomllib
import warnings

warnings.filterwarnings(action="ignore")


def _check_module(module_name: str) -> bool:
    """Check if a module can be imported.

    Parameters
    ----------
    module_name : str
        The name of the module.

    Returns
    -------
    bool
        ``True`` if the module can be imported, ``False`` otherwise.
    """
    try:
        importlib.import_module(name=module_name)

    except ImportError:
        print("not installed.")
        return False

    print("installed.")
    return True


def _check_exe(prg_name: str) -> bool:
    """Check if an executable is available in PATH.

    Parameters
    ----------
    prg_name : str
        The name of the program.

    Returns
    -------
    bool
        ``True`` if the executable is found, ``False`` otherwise.
    """
    prg = shutil.which(cmd=prg_name)
    if prg is None:
        print("not installed.")
        return False

    if os.access(path=prg, mode=os.X_OK) is False:
        print("accessible but not executable.")
        return False

    print("installed.")
    return True


def _check_qmdesc() -> bool:
    """Check qmdesc installation.

    Returns
    -------
    bool
        ``True`` if qmdesc is correctly installed, ``False`` otherwise.
    """
    from bonafide.features.qmdesc_ import get_qmdesc_handler

    try:
        handler = get_qmdesc_handler()

    except ImportError:
        print("not installed.")
        return False

    try:
        _ = handler.predict("CCO")

    except Exception as e:
        print(f"not installed correctly: {e}")
        return False

    print("installed.")
    return True


def _check_alfabet() -> bool:
    """Check alfabet installation.

    The Python interpreter of the ALFABET environment is taken from the
    ``BONAFIDE_ALFABET_PYTHON`` environment variable or, if it is not set, from the
    ``_feature_config.toml`` file.

    Returns
    -------
    bool
        ``True`` if alfabet is correctly installed, ``False`` otherwise.
    """
    interpreter_path = os.environ.get("BONAFIDE_ALFABET_PYTHON")

    if not interpreter_path:
        config_path = os.path.join(os.path.dirname(__file__), "_feature_config.toml")
        with open(config_path, "rb") as config_file:
            configs = tomllib.load(config_file)

        interpreter_path = configs.get("alfabet", {}).get("python_interpreter_path", None)

    if interpreter_path is None:
        print(
            "not installed (Python interpreter not found in BONAFIDE_ALFABET_PYTHON or "
            "'_feature_config.toml' (alfabet.python_interpreter_path))."
        )
        return False

    interpreter_path = os.path.expanduser(path=interpreter_path)
    if os.path.isfile(interpreter_path) is False:
        print(f"not installed (Python interpreter path is invalid, {interpreter_path}).")
        return False

    command = f'"{interpreter_path}" -c "import alfabet"'
    exit_code = os.system(command=command)
    if exit_code != 0:
        print("not installed.")
        return False

    print(f"installed (in external environment at {interpreter_path}).")
    return True


def run_checks() -> bool:
    """Run all checks and print the results.

    Returns
    -------
    bool
        ``True`` if all checks pass, ``False`` otherwise.
    """
    modules = ["morfeus", "mendeleev", "rdkit", "psi4", "dftd3", "dbstep", "dscribe"]
    executables = ["kallisto", "Multiwfn_noGUI", "xtb"]
    checks = sorted(modules + executables + ["qmdesc"], key=str.lower)
    checks.append("alfabet")

    counter = 0
    for check in checks:
        print(f"> {check:17} ...   ", end=" ")
        if check in modules:
            passed = _check_module(module_name=check)
        elif check in executables:
            passed = _check_exe(prg_name=check)
        elif check == "qmdesc":
            passed = _check_qmdesc()
        else:
            passed = _check_alfabet()

        if passed is True:
            counter += 1

    return counter == len(checks)


def main() -> int:
    """Entry point of the ``bonafide-check`` command.

    Returns
    -------
    int
        The exit code, ``0`` if all checks pass, ``1`` otherwise.
    """
    import bonafide

    print("--------------------------")
    print("BONAFIDE dependency checks")
    print("--------------------------")
    print(f"BONAFIDE version: {bonafide.__version__}\n")
    if run_checks() is True:
        print("\n==> Hurray, all dependencies are installed correctly!\n")
        return 0

    print("\n==> Caution, some dependencies are missing or not correctly installed.\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
