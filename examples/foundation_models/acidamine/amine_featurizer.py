"""BONAFIDE side of the AcidAmine 2D primary amine example.

Usage (in the BONAFIDE environment):

.. code:: python

   from bonafide import AtomBondFeaturizer
   from amine_featurizer import AcidAmineAmine2D

   f = AtomBondFeaturizer()
   added = f.add_foundation_model(AcidAmineAmine2D)
   f.set_options(
       configs=[
           ("acidamine.python", "/path/to/acidamine-env/bin/python"),
           ("acidamine.options.repo_dir", "/path/to/AcidAmine_Descriptor_Predict"),
           ("acidamine.options.models_dir", "/path/to/2D/trained_models"),
       ]
   )
   f.read_input(input_value="NCC1CCCCC1", namespace="amine")
   f.featurize_atoms(atom_indices="all", feature_indices=list(added.values())[:24])
"""

import os

from bonafide.foundation import FoundationModelFeaturizer, Output

_HERE = os.path.dirname(os.path.abspath(__file__))

STATISTICS = ["boltz", "max", "min", "min_e"]

# Atom properties; each is predicted for the amine N (N1) and/or its alpha carbon (C2)
ATOM_PROPERTIES = [
    "nbo_charge",
    "nmr_shift",
    "pyrd_ar",
    "nbo_lp_energy",
    "nbo_lp_occupancy",
    "v_bur",
]

# Properties of the N1-C2 bond
BOND_PROPERTIES = ["sterimol_b1", "sterimol_b5", "sterimol_l"]


class AcidAmineAmine2D(FoundationModelFeaturizer):
    """Conformer-ensemble DFT descriptors of primary amines predicted from SMILES with the 2D
    GINE models of AcidAmine_Descriptor_Predict. Only defined for the amine N, its alpha carbon,
    and the bond between them; all other atoms and bonds are "_inaccessible"."""

    origin = "acidamine"
    input_mode = "2D"
    encoder = os.path.join(_HERE, "amine_encoder.py") + ":AmineEncoder"
    options = {"repo_dir": "", "models_dir": ""}
    outputs = [
        Output(f"{prop}_{stat}", "atom", indices_key=f"{prop}_{stat}_atoms")
        for stat in STATISTICS
        for prop in ATOM_PROPERTIES
    ] + [
        Output(f"{prop}_{stat}", "bond", indices_key="n1_c2_bond")
        for stat in STATISTICS
        for prop in BOND_PROPERTIES
    ]
