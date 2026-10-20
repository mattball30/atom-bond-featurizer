"""Encoder base class and worker loop for foundation model featurizers.

This module is executed with the Python interpreter of the foundation model's environment, in
which BONAFIDE is usually NOT installed. Therefore, it must only depend on the Python standard
library. The encoder module written by the user imports the base class from here:

.. code:: python

   from bonafide_encoder import EncoderError, FoundationModelEncoder, MolInput

This works because the worker loop registers this module as ``bonafide_encoder`` in
``sys.modules`` before the user's encoder module is imported. The directory of this file is not
added to ``sys.path``, so that the other BONAFIDE modules next to it cannot shadow modules of the
foundation model's environment.

Communication with BONAFIDE happens through JSON lines on stdin and stdout. To protect this channel
from anything the foundation model prints, the original stdout is reserved for the protocol and
everything else written to stdout is redirected to stderr.
"""

import argparse
import importlib.util
import json
import math
import os
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


class EncoderError(Exception):
    """Raise this exception in ``encode()`` to report a handled error for a molecule (e.g.,
    "molecule outside of the model's domain"). Only the message is passed on to BONAFIDE, no
    traceback.
    """


class MolInput:
    """The molecule data sent from BONAFIDE to the encoder.

    Attributes
    ----------
    smiles : str
        The canonical SMILES string of the molecule (without hydrogen atoms). Do not
        re-canonicalize it. The atom order of ``Chem.MolFromSmiles(smiles)`` (followed by
        ``Chem.AddHs()`` if ``has_explicit_hydrogens`` is ``True``) is the "canonical" atom order.
    molblock : str
        The MDL mol block of the molecule in BONAFIDE's atom order (the "input" atom order). It
        contains the coordinates for 3D input.
    atomic_numbers : List[int]
        The atomic numbers in the input atom order.
    coordinates : Optional[List[List[float]]]
        The cartesian coordinates in Angstrom in the input atom order; ``None`` for 2D input.
    bonds : List[Tuple[int, int, float]]
        The bonds as (begin atom index, end atom index, bond order) in the input atom and bond
        order.
    charge : int
        The total charge of the molecule.
    multiplicity : Optional[int]
        The multiplicity of the molecule; ``None`` if not set in BONAFIDE.
    has_explicit_hydrogens : bool
        Whether the molecule in BONAFIDE contains explicit hydrogen atoms.
    input_mode : str
        Either "2D" or "3D".
    """

    smiles: str
    molblock: str
    atomic_numbers: List[int]
    coordinates: Optional[List[List[float]]]
    bonds: List[Tuple[int, int, float]]
    charge: int
    multiplicity: Optional[int]
    has_explicit_hydrogens: bool
    input_mode: str

    def __init__(self, **data: Any) -> None:
        for key, value in data.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"MolInput(smiles={self.smiles!r}, input_mode={self.input_mode!r})"


class FoundationModelEncoder:
    """Base class for foundation model encoders.

    Child classes implement ``encode()`` and optionally ``load()``. ``encode()`` returns a
    dictionary of named outputs. Each output is a (nested) list or anything with a ``tolist()``
    method (e.g., a numpy array or a torch tensor). How these outputs are turned into BONAFIDE
    features is defined on the BONAFIDE side with ``Output`` objects.
    """

    def load(self, **options: Any) -> None:
        """Load the model. Called once when the worker starts.

        Parameters
        ----------
        **options : Any
            The options defined in the ``options`` attribute of the BONAFIDE featurizer class
            (changeable with ``set_options()``).

        Returns
        -------
        None
        """

    def encode(self, mol: MolInput) -> Dict[str, Any]:
        """Run the model for a single molecule.

        Parameters
        ----------
        mol : MolInput
            The molecule data.

        Returns
        -------
        Dict[str, Any]
            The named model outputs.
        """
        raise NotImplementedError("encode() must be implemented in the encoder class.")


def _to_jsonable(value: Any) -> Any:
    """Convert model outputs (numpy arrays, tensors, numpy scalars) to JSON-serializable data."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _load_encoder_class(spec: str) -> Any:
    """Import the encoder class from a "path/to/file.py:ClassName" specification."""
    file_path, _, class_name = spec.rpartition(":")
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    sys.path.insert(0, os.path.dirname(os.path.abspath(file_path)))
    module_spec = importlib.util.spec_from_file_location(module_name, file_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import '{file_path}'")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return getattr(module, class_name)


def _error_response(e: BaseException) -> Dict[str, Any]:
    if isinstance(e, EncoderError):
        return {"ok": False, "error": str(e), "handled": True}
    return {"ok": False, "error": traceback.format_exc(), "handled": False}


def main(argv: Optional[List[str]] = None) -> None:
    """Run the worker loop: load the encoder, then answer encode requests until stdin closes."""
    parser = argparse.ArgumentParser(description="BONAFIDE foundation model worker.")
    parser.add_argument("--encoder", required=True, help="path/to/file.py:ClassName")
    args = parser.parse_args(argv)

    # Reserve the original stdout for the protocol and send all other output to stderr
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    # Make "from bonafide_encoder import ..." in the encoder module refer to this module (and not
    # to a second copy of it), so that EncoderError raised there is caught here
    sys.modules.setdefault("bonafide_encoder", sys.modules[__name__])

    def _send(message: Dict[str, Any]) -> None:
        protocol.write(json.dumps(message) + "\n")
        protocol.flush()

    encoder = None
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            if request["cmd"] == "load":
                encoder = _load_encoder_class(args.encoder)()
                encoder.load(**request.get("options", {}))
                response: Dict[str, Any] = {"ok": True}
            elif request["cmd"] == "encode":
                if encoder is None:
                    raise RuntimeError("encoder was not loaded")
                outputs = encoder.encode(MolInput(**request["mol"]))
                if not isinstance(outputs, dict):
                    raise TypeError(
                        f"encode() must return a dict but returned {type(outputs).__name__}"
                    )
                response = {"ok": True, "outputs": _to_jsonable(outputs)}
            else:
                raise ValueError(f"unknown command '{request['cmd']}'")
        except Exception as e:
            response = _error_response(e)
        _send(response)


if __name__ == "__main__":
    main()
