"""Base feature factory for foundation models that run in an external Python environment."""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

from rdkit import Chem

from bonafide.utils.foundation_model_worker import get_worker
from bonafide.utils.base_featurizer import BaseFeaturizer
from bonafide.utils.helper_functions_chemistry import (
    get_atom_bond_mapping_dicts,
    get_charge_from_mol_object,
)

_STORE_DATA_TYPES = {"scalar": ["float", "int", "bool"], "vector": ["str"]}


class Output:
    """Mapping of one raw encoder output onto one BONAFIDE atom or bond feature.

    Parameters
    ----------
    key : str
        The name of the output in the dictionary returned by the encoder's ``encode()`` method.
    feature_type : str
        Either "atom" or "bond".
    name : Optional[str], optional
        The feature name; the full BONAFIDE feature name is ``{origin}{2D|3D}-{feature_type}-{name}``.
        Defaults to ``key``.
    store : str, optional
        "scalar" (one number per atom or bond) or "vector" (one list of numbers per atom or bond,
        stored as comma-separated string). By default "scalar".
    index : Optional[int], optional
        For vector-valued raw outputs, store only this component as scalar feature.
    data_type : Optional[str], optional
        The data type of scalar features ("float", "int", or "bool"); defaults to "float" for
        scalars and "str" for vectors.
    atom_order : Optional[str], optional
        The atom order the output refers to: "input" (the atom order of ``MolInput.molblock``)
        or "canonical" (the atom order of ``MolInput.smiles``). Defaults to "canonical" for 2D and
        "input" for 3D foundation models.
    indices_key : Optional[str], optional
        The name of an encoder output that identifies the atoms or bonds the values belong to.
        For atom features, it must contain atom indices; for bond features, pairs of atom
        indices. If not given, the output must contain one value for every atom (in atom order)
        or bond (in bond order).
    nan : str, optional
        How missing values (NaN or ``None``) are treated: "missing" (the feature is
        "_inaccessible" for that atom or bond) or "error". By default "missing".
    """

    def __init__(
        self,
        key: str,
        feature_type: str,
        name: Optional[str] = None,
        store: str = "scalar",
        index: Optional[int] = None,
        data_type: Optional[str] = None,
        atom_order: Optional[str] = None,
        indices_key: Optional[str] = None,
        nan: str = "missing",
    ) -> None:
        self.key = key
        self.feature_type = str(feature_type).strip().lower()
        self.name = str(name if name is not None else key).strip().lower()
        self.store = str(store).strip().lower()
        self.index = index
        self.data_type = (
            data_type
            if data_type is not None
            else _STORE_DATA_TYPES[self.store if self.store in _STORE_DATA_TYPES else "scalar"][0]
        )
        self.atom_order = atom_order
        self.indices_key = indices_key
        self.nan = str(nan).strip().lower()

        if self.feature_type not in ["atom", "bond"]:
            raise ValueError(f"Output '{key}': feature_type must be 'atom' or 'bond'.")
        if self.store not in _STORE_DATA_TYPES:
            raise ValueError(f"Output '{key}': store must be 'scalar' or 'vector'.")
        if self.data_type not in _STORE_DATA_TYPES[self.store]:
            raise ValueError(
                f"Output '{key}': data_type for store='{self.store}' must be one of "
                f"{_STORE_DATA_TYPES[self.store]}."
            )
        if self.index is not None and (self.store != "scalar" or not isinstance(index, int)):
            raise ValueError(f"Output '{key}': index must be an int and requires store='scalar'.")
        if self.atom_order not in [None, "input", "canonical"]:
            raise ValueError(f"Output '{key}': atom_order must be 'input' or 'canonical'.")
        if self.nan not in ["missing", "error"]:
            raise ValueError(f"Output '{key}': nan must be 'missing' or 'error'.")
        if len(self.name) == 0 or " " in self.name:
            raise ValueError(f"Output '{key}': name must be non-empty and contain no spaces.")

    def __repr__(self) -> str:
        return f"Output({self.key!r}, feature_type={self.feature_type!r}, name={self.name!r})"


class FoundationModelFeaturizer(BaseFeaturizer):
    """Base feature factory for foundation models running in an external Python environment.

    Subclass it, set the class attributes below, and register the subclass with
    ``AtomBondFeaturizer.add_foundation_model()``. One feature is registered per ``Output``.

    Attributes
    ----------
    origin : str
        The name of the foundation model, used as origin and configuration section.
    input_mode : str
        "2D": the encoder receives the molecule as canonical SMILES string (the model may still
        generate 3D structures internally). "3D": the encoder additionally receives the
        coordinates of each conformer.
    encoder : str
        The encoder class as "path/to/encoder_file.py:ClassName". The class must inherit from
        ``FoundationModelEncoder`` (``bonafide_encoder.py``).
    outputs : List[Output]
        The mapping of the raw encoder outputs onto BONAFIDE features.
    python : Optional[str]
        The Python interpreter of the foundation model environment. If ``None``, the
        ``BONAFIDE_<ORIGIN>_PYTHON`` environment variable is used. Changeable with
        ``set_options("<origin>.python", ...)``.
    options : Dict[str, Any]
        Options passed to the encoder's ``load()`` method. Changeable with
        ``set_options("<origin>.options.<key>", ...)``.
    timeout : float
        Timeout in seconds for loading the model and for each encode request.
    requires_charge : bool
        Whether the charge of the molecule must be set in BONAFIDE.
    requires_multiplicity : bool
        Whether the multiplicity of the molecule must be set in BONAFIDE.
    """

    origin: str
    input_mode: str
    encoder: str
    outputs: List[Output] = []
    python: Optional[str] = None
    options: Dict[str, Any] = {}
    timeout: float = 600.0
    requires_charge: bool = False
    requires_multiplicity: bool = False

    def __init__(self) -> None:
        self.extraction_mode = "multi"
        super().__init__()

    @classmethod
    def feature_name_of(cls, output: Output) -> str:
        """Get the full BONAFIDE feature name of an output."""
        return f"{cls.origin}{cls.input_mode}-{output.feature_type}-{output.name}"

    def calculate(self) -> None:
        """Run the encoder for the current conformer and map its outputs onto all features of this
        foundation model with the requested feature type."""
        mapping_dict_atoms, mapping_dict_bonds, canonical_smiles = get_atom_bond_mapping_dicts(
            self.mol
        )

        python = self.python or os.environ.get(f"BONAFIDE_{self.origin.upper()}_PYTHON")
        if not python:
            self._err = (
                f"no Python interpreter is defined for the '{self.origin}' foundation model; set "
                f"it with set_options(('{self.origin}.python', '/path/to/env/bin/python')) or "
                f"with the BONAFIDE_{self.origin.upper()}_PYTHON environment variable"
            )
            return

        worker = get_worker(
            python=os.path.expanduser(python),
            encoder=self.encoder,
            options=self.options,
            timeout=float(self.timeout),
        )
        raw, err = worker.encode(self._get_mol_input(canonical_smiles=canonical_smiles))
        if err is not None:
            self._err = err
            return
        assert raw is not None  # for type checker

        self._output_errors: Dict[str, str] = {}
        try:
            self.results = self.map_outputs(
                raw=raw,
                mapping_dict_atoms=mapping_dict_atoms,
                mapping_dict_bonds=mapping_dict_bonds,
            )
        except ValueError as e:
            self._err = f"mapping the outputs of the '{self.origin}' foundation model failed: {e}"
            return

        # Errors of individual outputs only affect the respective feature
        if self.feature_name in self._output_errors:
            self._err = (
                f"mapping the outputs of the '{self.origin}' foundation model failed: "
                f"{self._output_errors[self.feature_name]}"
            )

    def map_outputs(
        self,
        raw: Dict[str, Any],
        mapping_dict_atoms: Dict[int, int],
        mapping_dict_bonds: Dict[int, int],
    ) -> Dict[int, Dict[str, Any]]:
        """Map the raw encoder outputs onto the features of the requested feature type.

        Override this method if the ``Output`` declarations cannot express the mapping. It must
        return a dictionary mapping the atom or bond indices (BONAFIDE order) to dictionaries of
        feature names and values for all features of the requested feature type. Raising
        ``ValueError`` marks the calculation of all of these features as failed.

        Parameters
        ----------
        raw : Dict[str, Any]
            The raw outputs of the encoder.
        mapping_dict_atoms : Dict[int, int]
            Maps canonical atom indices to input atom indices.
        mapping_dict_bonds : Dict[int, int]
            Maps canonical bond indices to input bond indices.

        Returns
        -------
        Dict[int, Dict[str, Any]]
            The feature values; missing values are set to "_inaccessible".
        """
        n_atoms = self.mol.GetNumAtoms()
        n_bonds = self.mol.GetNumBonds()
        outputs = [o for o in self.outputs if o.feature_type == self.feature_type]
        n = n_atoms if self.feature_type == "atom" else n_bonds

        # Every feature of this type gets a value for every atom or bond, so that the cache is
        # filled completely after one encoder call
        results: Dict[int, Dict[str, Any]] = {
            idx: {self.feature_name_of(o): "_inaccessible" for o in outputs} for idx in range(n)
        }

        for o in outputs:
            feature_name = self.feature_name_of(o)
            try:
                values = self._map_output(
                    o=o,
                    raw=raw,
                    mapping_dict_atoms=mapping_dict_atoms,
                    mapping_dict_bonds=mapping_dict_bonds,
                )
            except ValueError as e:
                # The feature is removed from the results so that it is not cached
                self._output_errors[feature_name] = str(e)
                for data in results.values():
                    del data[feature_name]
                continue
            for target, value in values.items():
                results[target][feature_name] = value

        return results

    def _map_output(
        self,
        o: Output,
        raw: Dict[str, Any],
        mapping_dict_atoms: Dict[int, int],
        mapping_dict_bonds: Dict[int, int],
    ) -> Dict[int, Any]:
        """Map a single raw output onto the input atom or bond indices."""
        if o.key not in raw:
            raise ValueError(f"the encoder did not return the output '{o.key}'")
        values = raw[o.key]
        if not isinstance(values, list):
            raise ValueError(f"output '{o.key}' must be a list or array")

        canonical = (o.atom_order or ("canonical" if self.input_mode == "2D" else "input")) == (
            "canonical"
        )
        targets = self._get_target_indices(
            o=o,
            raw=raw,
            n_values=len(values),
            canonical=canonical,
            mapping_dict_atoms=mapping_dict_atoms,
            mapping_dict_bonds=mapping_dict_bonds,
        )

        mapped = {}
        for target, value in zip(targets, values):
            if target is None:
                continue  # atom or bond not present in the input molecule
            mapped[target] = self._convert_value(o=o, value=value)
        return mapped

    def _get_target_indices(
        self,
        o: Output,
        raw: Dict[str, Any],
        n_values: int,
        canonical: bool,
        mapping_dict_atoms: Dict[int, int],
        mapping_dict_bonds: Dict[int, int],
    ) -> List[Optional[int]]:
        """Get the input atom or bond index for each value of an output (``None`` if the atom or
        bond does not exist in the input molecule)."""

        def _atom(idx: Any) -> Optional[int]:
            idx = int(idx)
            if canonical:
                return mapping_dict_atoms.get(idx)
            return idx if 0 <= idx < self.mol.GetNumAtoms() else None

        # Values for all atoms or bonds in order
        if o.indices_key is None:
            if o.feature_type == "atom":
                n_expected = len(mapping_dict_atoms) if canonical else self.mol.GetNumAtoms()
                if n_values != n_expected:
                    raise ValueError(
                        f"output '{o.key}' has {n_values} entries but the molecule has "
                        f"{n_expected} atoms; use indices_key for outputs that only cover some "
                        "atoms"
                    )
                return [_atom(i) for i in range(n_values)]
            n_expected = len(mapping_dict_bonds) if canonical else self.mol.GetNumBonds()
            if n_values != n_expected:
                raise ValueError(
                    f"output '{o.key}' has {n_values} entries but the molecule has {n_expected} "
                    "bonds; use indices_key for outputs that only cover some bonds"
                )
            if canonical:
                return [mapping_dict_bonds.get(i) for i in range(n_values)]
            return list(range(n_values))

        # Values for the atoms or bonds given by indices_key
        if o.indices_key not in raw:
            raise ValueError(f"the encoder did not return the indices output '{o.indices_key}'")
        indices = raw[o.indices_key]
        if not isinstance(indices, list) or len(indices) != n_values:
            raise ValueError(
                f"indices output '{o.indices_key}' must be a list with one entry per value of "
                f"'{o.key}'"
            )
        if o.feature_type == "atom":
            return [_atom(i) for i in indices]

        targets: List[Optional[int]] = []
        for pair in indices:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(
                    f"indices output '{o.indices_key}' must contain pairs of atom indices"
                )
            a, b = _atom(pair[0]), _atom(pair[1])
            bond = None
            if a is not None and b is not None:
                bond = self.mol.GetBondBetweenAtoms(a, b)
            targets.append(bond.GetIdx() if bond is not None else None)
        return targets

    @staticmethod
    def _convert_value(o: Output, value: Any) -> Any:
        """Convert a raw value to the feature value."""
        if o.index is not None:
            if not isinstance(value, list) or not -len(value) <= o.index < len(value):
                raise ValueError(f"output '{o.key}' has no component with index {o.index}")
            value = value[o.index]

        if o.store == "vector":
            if not isinstance(value, list):
                value = [value]
            if any(_is_missing(v) for v in value):
                if o.nan == "error":
                    raise ValueError(f"output '{o.key}' contains missing values")
                return "_inaccessible"
            return ",".join(str(float(v)) for v in value)

        if isinstance(value, list):
            raise ValueError(
                f"output '{o.key}' is vector-valued; use store='vector' or select a component "
                "with index"
            )
        if _is_missing(value):
            if o.nan == "error":
                raise ValueError(f"output '{o.key}' contains missing values")
            return "_inaccessible"
        if o.data_type == "int":
            return int(value)
        if o.data_type == "bool":
            return bool(value)
        return float(value)

    def _get_mol_input(self, canonical_smiles: str) -> Dict[str, Any]:
        """Collect the molecule data sent to the encoder (see ``MolInput``)."""
        mol = self.mol
        coordinates: Optional[List[List[float]]] = None
        if self.input_mode == "3D":
            coordinates = [[float(x) for x in pos] for pos in mol.GetConformer(0).GetPositions()]

        bonds: List[Tuple[int, int, float]] = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble())
            for b in mol.GetBonds()
        ]

        # The molblock is only used for the atom and bond data in 2D mode, so the conformer is
        # removed to make the input identical for all conformers of an ensemble
        mol_for_block = Chem.Mol(mol)
        if self.input_mode == "2D":
            mol_for_block.RemoveAllConformers()

        charge = self.charge if self.charge is not None else get_charge_from_mol_object(mol)

        return {
            "smiles": canonical_smiles,
            "molblock": Chem.MolToMolBlock(mol_for_block),
            "atomic_numbers": [a.GetAtomicNum() for a in mol.GetAtoms()],
            "coordinates": coordinates,
            "bonds": bonds,
            "charge": int(charge),
            "multiplicity": self.multiplicity,
            "has_explicit_hydrogens": any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()),
            "input_mode": self.input_mode,
        }


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
