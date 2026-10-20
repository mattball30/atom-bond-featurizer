"""Validation of foundation model featurizer classes before they are added to BONAFIDE."""

import json
import os
import re
from typing import Any, Dict, List, Tuple, Type

from bonafide.foundation.featurizer import FoundationModelFeaturizer, Output
from bonafide.utils.feature_factories import FEATURE_FACTORIES


def _camel(s: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[_\-]", s))


def foundation_model_data_validator(
    model: Any, feature_config: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """Validate a foundation model featurizer class and generate the feature metadata.

    Parameters
    ----------
    model : Any
        The foundation model featurizer class.
    feature_config : Dict[str, Any]
        The current configuration settings (used to check that the origin is not in use).

    Returns
    -------
    Tuple[str, List[Dict[str, Any]], Dict[str, Any]]
        A tuple containing:

        * The origin of the foundation model.
        * The feature metadata for each output (in the format of ``_feature_info.json``, with
          the factory class instead of its name).
        * The configuration settings of the foundation model.
    """
    if not isinstance(model, type) or not issubclass(model, FoundationModelFeaturizer):
        raise TypeError(
            "Invalid input to 'model': must be a class inheriting from FoundationModelFeaturizer."
        )

    # Origin
    origin = str(getattr(model, "origin", "")).strip().lower()
    if re.fullmatch(r"[a-z][a-z0-9_]*", origin) is None:
        raise ValueError(
            f"Invalid 'origin' of '{model.__name__}': '{origin}'; it must start with a letter and "
            "contain only letters, digits, and underscores."
        )
    if origin in feature_config:
        raise ValueError(f"Invalid 'origin' of '{model.__name__}': '{origin}' is already in use.")

    # Input mode
    input_mode = str(getattr(model, "input_mode", "")).strip().upper()
    if input_mode not in ["2D", "3D"]:
        raise ValueError(
            f"Invalid 'input_mode' of '{model.__name__}': must be '2D' or '3D' but obtained "
            f"'{input_mode}'."
        )

    # Encoder
    encoder = getattr(model, "encoder", None)
    if not isinstance(encoder, str) or ":" not in encoder:
        raise ValueError(
            f"Invalid 'encoder' of '{model.__name__}': must be a string of the form "
            "'path/to/encoder_file.py:ClassName'."
        )
    encoder_file, _, encoder_class = encoder.rpartition(":")
    encoder_file = os.path.abspath(os.path.expanduser(encoder_file))
    if not os.path.isfile(encoder_file):
        raise ValueError(
            f"Invalid 'encoder' of '{model.__name__}': file '{encoder_file}' does not exist."
        )
    if not encoder_class.isidentifier():
        raise ValueError(
            f"Invalid 'encoder' of '{model.__name__}': '{encoder_class}' is not a valid class name."
        )

    # Python interpreter
    python = model.python
    if python is not None:
        if not isinstance(python, str):
            raise TypeError(f"Invalid 'python' of '{model.__name__}': must be a str or None.")
        python = os.path.abspath(os.path.expanduser(python))
        if not os.path.isfile(python):
            raise ValueError(
                f"Invalid 'python' of '{model.__name__}': file '{python}' does not exist."
            )

    # Options and timeout
    if not isinstance(model.options, dict):
        raise TypeError(f"Invalid 'options' of '{model.__name__}': must be a dict.")
    try:
        json.dumps(model.options)
    except (TypeError, ValueError):
        raise TypeError(f"Invalid 'options' of '{model.__name__}': must be JSON-serializable.")
    if isinstance(model.timeout, bool) or not isinstance(model.timeout, (int, float)):
        raise TypeError(f"Invalid 'timeout' of '{model.__name__}': must be a number.")
    if model.timeout <= 0:
        raise ValueError(f"Invalid 'timeout' of '{model.__name__}': must be positive.")
    for attr in ["requires_charge", "requires_multiplicity"]:
        if not isinstance(getattr(model, attr), bool):
            raise TypeError(f"Invalid '{attr}' of '{model.__name__}': must be a bool.")

    # Outputs
    outputs = model.outputs
    if not isinstance(outputs, list) or len(outputs) == 0:
        raise ValueError(f"Invalid 'outputs' of '{model.__name__}': must be a non-empty list.")
    seen = set()
    for o in outputs:
        if not isinstance(o, Output):
            raise TypeError(
                f"Invalid 'outputs' of '{model.__name__}': all entries must be Output objects."
            )
        if (o.feature_type, o.name) in seen:
            raise ValueError(
                f"Invalid 'outputs' of '{model.__name__}': the {o.feature_type} feature name "
                f"'{o.name}' is used more than once."
            )
        seen.add((o.feature_type, o.name))

    # One factory class per feature; they share the class attributes of the model class
    metadata_list = []
    for o in outputs:
        factory_name = f"{_camel(origin)}{input_mode}{o.feature_type.capitalize()}{_camel(o.name)}"
        # Factories generated for foundation models may be replaced, e.g., when the model is added
        # to a new AtomBondFeaturizer instance in the same session
        existing = FEATURE_FACTORIES.get(factory_name)
        if existing is not None and getattr(existing, "_is_generated_factory", False) is False:
            raise ValueError(
                f"Invalid 'outputs' of '{model.__name__}': the factory class name "
                f"'{factory_name}' is already in use."
            )
        factory: Type[FoundationModelFeaturizer] = type(
            factory_name,
            (model,),
            {
                "origin": origin,
                "input_mode": input_mode,
                "_is_generated_factory": True,
                "__module__": model.__module__,
            },
        )
        metadata_list.append(
            {
                "name": factory.feature_name_of(o),
                "origin": origin,
                "feature_type": o.feature_type,
                "dimensionality": input_mode,
                "data_type": o.data_type,
                "requires_electronic_structure_data": False,
                "requires_bond_data": True,
                "requires_charge": model.requires_charge,
                "requires_multiplicity": model.requires_multiplicity,
                "config_path": origin,
                "factory": factory,
            }
        )

    config = {
        "python": python,
        "encoder": f"{encoder_file}:{encoder_class}",
        "options": dict(model.options),
        "timeout": float(model.timeout),
    }
    return origin, metadata_list, config
