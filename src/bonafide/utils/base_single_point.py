"""Base class for single-point energy calculations with different computational engines."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from bonafide.utils.base_mixin import _BaseMixin
from bonafide.utils.helper_functions import get_function_or_method_name

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from bonafide.utils.molecule_vault import MolVault


class BaseSinglePoint(_BaseMixin):
    """Run single-point energy calculations with different computational engines.

    All conformers in the molecule vault are processed sequentially.

    Attributes
    ----------
    _keep_output_files : bool
        If ``True``, all output files created during the feature calculations are kept. If
        ``False``, they are removed when the calculation is done.
    charge : int
        The total charge of the molecule.
    conformer_name : str
        The name of the conformer.
    coordinates : NDArray[np.float64]
        The cartesian coordinates of the conformer.
    elements : NDArray[np.str\_]
        The element symbols of the molecule.
    engine_name : str
        The name of the computational engine (must be set in the child class).
    mol_vault : MolVault
        The dataclass for storing all relevant data on the molecule.
    multiplicity : int
        The spin multiplicity of the molecule.
    """

    _keep_output_files: bool
    charge: int
    conformer_name: str
    coordinates: NDArray[np.float64]
    elements: NDArray[np.str_]
    engine_name: str
    method: str
    mol_vault: MolVault
    multiplicity: int
    solvent: str
    state: str

    def __init__(self, **kwargs: Any) -> None:
        # Set all attributes required for the single-point energy calculation
        for attr_name, value in kwargs.items():
            setattr(self, attr_name, value)

        # Check if single-point energy class is correctly implemented
        self._check_requirements()

    def _check_requirements(self) -> None:
        """Check if the respective single-point energy class (child class) implements the
        ``calculate()`` method and sets the ``engine_name`` attribute.

        Returns
        -------
        None
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Check if child class has mandatory calculate() method
        method_names = [
            attr
            for attr in dir(self)
            if callable(getattr(self, attr)) is True and not attr.startswith("__")
        ]
        if "calculate" not in method_names:
            _errmsg = (
                "calculate() method must be implemented in engine-specific single-point "
                f"energy class '{self.__class__.__name__}'."
            )
            logging.error(f"'None' | {_loc}()\n{_errmsg}")
            raise NotImplementedError(_errmsg)

        # Check if child class sets engine_name attribute
        if "engine_name" not in vars(self):
            _errmsg = (
                "Attribute 'engine_name' must be set in engine-specific single-point energy class "
                f"'{self.__class__.__name__}'."
            )
            logging.error(f"'None' | {_loc}()\n{_errmsg}")
            raise AttributeError(f"{_loc}(): {_errmsg}")

    def run(
        self, state: str, stop_on_error: bool, write_el_struc_file: bool = True
    ) -> Tuple[List[Tuple[Optional[float], str]], List[Optional[str]], Optional[RuntimeError]]:
        """Run a single-point energy calculation for all conformers of the molecule in the
        molecule vault.

        Parameters
        ----------
        state : str
            The redox state of the molecule to consider, either "n", "n+1", or "n-1".
        stop_on_error : bool
            If ``True``, the loop over the conformers is stopped as soon as a calculation fails
            for a conformer. The failed conformer and all following conformers get ``None`` as
            energy and electronic structure data, and the error is returned so that the caller
            can raise it after the results were attached. If ``False``, the error is logged and the
            calculation continues with the next conformer.
        write_el_struc_file : bool, optional
            Whether to write the calculated electronic structure of the molecule to an electronic
            structure data file, by default ``True``.

        Returns
        -------
        Tuple[List[Tuple[Optional[float], str]], List[Optional[str]], Optional[RuntimeError]]
            A tuple containing:

            * A list of tuples with the electronic energy in kJ/mol (value, unit pair) for each
              conformer. In case the calculation failed or was not run, the energy is ``None``.
            * A list of paths to the electronic structure data files for each conformer. If they
              were not requested or the calculation failed or was not run, the paths are
              ``None``.
            * The error that stopped the calculation if ``stop_on_error`` is ``True`` and a
              calculation failed (chained to the original exception), otherwise ``None``.
        """
        _loc = f"{self.__class__.__name__}.{get_function_or_method_name()}"

        # Define additional attributes
        # It is ensured that the molecule vault contains the required data for the single-point
        # energy calculation.
        self.elements = self.mol_vault.elements  # type: ignore[assignment]
        self.charge = self.mol_vault.charge  # type: ignore[assignment]
        self.multiplicity = self.mol_vault.multiplicity  # type: ignore[assignment]

        energies: List[Tuple[Optional[float], str]] = []
        electronic_strucs: List[Optional[str]] = []
        stop_error: Optional[RuntimeError] = None

        # Loop over all conformers in the molecule vault
        for conf_idx, mol in enumerate(self.mol_vault.mol_objects):
            # Setup up working directory and change to it
            self.conformer_name = self.mol_vault.conformer_names[conf_idx]
            _namespace = self.conformer_name[::-1].split("__", 1)[-1][::-1]
            self._setup_work_dir()

            # Initialize results as None
            energy = None
            molden_file_path = None
            error: Optional[Exception] = None

            # Skip conformers that were labeled as invalid
            if self.mol_vault.is_valid[conf_idx] is False and state == "n":
                logging.warning(
                    f"'{_namespace}' | {_loc}()\nSkipping conformer with index {conf_idx} for "
                    f"single-point energy calculation for state '{state}' with {self.engine_name} "
                    "because it is invalid."
                )

            else:
                logging.info(
                    (
                        f"'{_namespace}' | {_loc}()\nRunning {self.engine_name} single "
                        f"point-energy calculation for conformer with index {conf_idx} for "
                        f"state '{state}'."
                    )
                )

                self.coordinates = mol.GetConformer(0).GetPositions()

                # Try to run the calculation
                try:
                    # self._check_requirements() ensures that the child class implements the
                    # calculate() method; mypy does not recognize this, so we ignore the type
                    # error here
                    energy, molden_file_path = self.calculate(  # type: ignore[attr-defined]
                        write_el_struc_file=write_el_struc_file
                    )
                except Exception as e:
                    error = e
                    _errmsg = (
                        f"Single-point energy calculation with {self.engine_name} failed for "
                        f"conformer with index {conf_idx} for state '{state}': {str(e)}."
                    )
                    if _errmsg.endswith(".."):
                        _errmsg = _errmsg[:-1]
                    if stop_on_error is True:
                        _errmsg += (
                            " The calculation was stopped: the remaining conformers were not "
                            "processed and, like the failed conformer, have no energy and "
                            "electronic structure data. This behavior can be changed by setting "
                            "the 'stop_on_error' parameter to False."
                        )
                    logging.error(f"'{_namespace}' | {_loc}()\n{_errmsg}")

            # Collect and store data
            energies.append((energy, "kj_mol"))
            electronic_strucs.append(molden_file_path)

            # Leave and clean up the working directory (also in case of an error)
            self._save_output_files()

            if error is not None and stop_on_error is True:
                stop_error = RuntimeError(f"{_loc}(): {_errmsg}")
                stop_error.__cause__ = error
                break

        # Fill in None for all conformers that were not processed
        _n_missing = len(self.mol_vault.mol_objects) - len(energies)
        energies.extend([(None, "kj_mol")] * _n_missing)
        electronic_strucs.extend([None] * _n_missing)

        return (energies, electronic_strucs, stop_error)
