"""Encoders for the 2D and 3D primary amine models of AcidAmine_Descriptor_Predict.

This file runs in the AcidAmine environment (*NOT* in the BONAFIDE environment). It wraps the
models from https://github.com/nsf-c-cas/AcidAmine_Descriptor_Predict (Digital Discovery, 2025,
4, 222), which predict conformer-ensemble DFT descriptors (Boltzmann-weighted, minimum, maximum,
and lowest-energy-conformer values) of specific atoms and bonds of primary amines:

* ``AmineEncoder2D``: the GINE models, which predict from the SMILES string.
* ``AmineEncoder3D``: the DimeNet++ models, which predict from a single 3D conformer. The authors
  generate their own RDKit conformer ensemble for each molecule and average the per-conformer
  predictions. Here, BONAFIDE sends each of its conformers; the predictions are made per conformer
  and can be averaged with BONAFIDE (``return_atom_features(reduce=True)``).

Each encoder runs in its own worker process, so the authors' ``utils`` module of the 2D or 3D
models is imported only by the encoder that needs it.

The models were trained for primary amines only (one -NH2 group bound to an sp3 carbon). Other
molecules are rejected with an EncoderError.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from bonafide_encoder import EncoderError, FoundationModelEncoder
from rdkit import Chem

# Atom properties of the 2D models: site of the functional group -> model folder (as in the
# authors' notebook)
ATOM_PROPERTIES_2D = {
    "N1_NBO_charge": "combined_amines",
    "N1_NMR_shift": "combined_amines",
    "N1_pyrd_Ar": "combined_amines",
    "N1_NBO_LP_energy": "combined_amines",
    "N1_NBO_LP_occupancy": "combined_amines",
    "N1_V_bur": "combined_amines",
    "C2_NBO_charge": "amines",
    "C2_NMR_shift": "amines",
}

# Atom properties of the amine N (N1) of the 3D models -> model folder
ATOM_PROPERTIES_3D = {
    "NBO_LP_energy": "combined_amines",
    "NBO_LP_occupancy": "combined_amines",
    "pyr_agranat": "combined_amines",
    "Vbur": "combined_amines",
}

# Bond properties of the N1-C2 bond (2D and 3D models)
BOND_PROPERTIES = ["Sterimol_B1", "Sterimol_B5", "Sterimol_L"]

STATISTICS = ["boltz", "max", "min", "min_E"]

# Output names of the 3D models; the same as for the 2D models where the descriptor is the same
OUTPUT_NAMES_3D = {
    "NBO_LP_energy": "nbo_lp_energy",
    "NBO_LP_occupancy": "nbo_lp_occupancy",
    "pyr_agranat": "pyrd_ar",
    "Vbur": "v_bur",
    "Sterimol_B1": "sterimol_b1",
    "Sterimol_B5": "sterimol_b5",
    "Sterimol_L": "sterimol_l",
}

# Primary amine bound to an sp3 carbon that is not part of an amide-like group
_AMINE_SMARTS = Chem.MolFromSmarts("[NX3;H2;!$(N[#6,#7,#16]=[O,S,N])][CX4]")


def _match_primary_amine(rdmol, smiles):
    """Return the (N1, C2) indices of the single primary amine group of a molecule."""
    matches = rdmol.GetSubstructMatches(_AMINE_SMARTS)
    if len(matches) != 1 or len(rdmol.GetSubstructMatches(Chem.MolFromSmarts("[NH2]C"))) != 1:
        raise EncoderError(
            "the AcidAmine amine models require exactly one primary amine (-NH2 bound to an "
            f"sp3 carbon) but found {len(matches)} in '{smiles}'",
        )
    return matches[0]


def _import_utils(repo_dir, model_type):
    """Import the authors' prediction code of the 2D or 3D models."""
    sys.path.insert(0, os.path.join(os.path.expanduser(repo_dir), model_type))
    import utils

    return utils


class AmineEncoder2D(FoundationModelEncoder):
    def load(self, repo_dir, models_dir, statistics=STATISTICS):
        """Load all 2D models.

        Parameters
        ----------
        repo_dir : str
            Path to the clone of the AcidAmine_Descriptor_Predict repository.
        models_dir : str
            Path to the "trained_models" directory extracted from 2D.zip (Figshare).
        statistics : List[str]
            The ensemble statistics to predict ("boltz", "max", "min", "min_E").
        """
        self.utils = _import_utils(repo_dir, "2D")
        self.statistics = list(statistics)
        models_dir = os.path.expanduser(models_dir)

        self.models = {}
        for stat in self.statistics:
            for prop, folder in ATOM_PROPERTIES_2D.items():
                selection = ("pamine", "atom", prop, stat)
                path = os.path.join(models_dir, folder, prop, stat, "model_best.pt")
                self.models[selection] = self.utils.load_model(*selection, {selection: path})
            for prop in BOND_PROPERTIES:
                selection = ("pamine", "bond", prop, stat)
                path = os.path.join(models_dir, "amines", prop, stat, "model_best.pt")
                self.models[selection] = self.utils.load_model(*selection, {selection: path})

        self.atom_selection = {("pamine", "atom", p): p.split("_")[0] for p in ATOM_PROPERTIES_2D}
        self.atom_selection.update({("pamine", "bond", p): ("N1", "C2") for p in BOND_PROPERTIES})

    def encode(self, mol):
        # Do not re-canonicalize: the atom indices returned must refer to the atom order of
        # MolFromSmiles(mol.smiles) followed by AddHs (the "canonical" order in BONAFIDE)
        mol_h = Chem.AddHs(Chem.MolFromSmiles(mol.smiles))
        _match_primary_amine(mol_h, mol.smiles)
        n1, c2, h3, h4 = self.utils.get_NH2_idx(mol_h)

        # Same input as the authors' generate_dataframe(), but without re-canonicalization
        df = pd.DataFrame(
            {
                "mols": [mol_h],
                "mols_noHs": [Chem.RemoveHs(mol_h)],
                "N1": [n1],
                "C2": [c2],
                "H3": [h3],
                "H4": [h4],
            },
        )

        outputs = {"n1_c2_bond": [[n1, c2]]}
        for selection, model in self.models.items():
            _, property_type, prop, stat = selection
            pred = self.utils.make_predictions(
                df,
                model,
                selection,
                self.atom_selection,
                keep_explicit_hydrogens=True,
                remove_Hs_except_functional=True,
            )
            value = float(pred[0])
            if property_type == "atom":
                site, name = prop.split("_", 1)
                key = f"{name.lower()}_{stat.lower()}"
                outputs.setdefault(key, [])
                outputs.setdefault(f"{key}_atoms", [])
                outputs[key].append(value)
                outputs[f"{key}_atoms"].append(n1 if site == "N1" else c2)
            else:
                outputs[f"{prop.lower()}_{stat.lower()}"] = [value]

        return outputs


class AmineEncoder3D(FoundationModelEncoder):
    def load(self, repo_dir, models_dir, statistics=STATISTICS):
        """Load all 3D models.

        Parameters
        ----------
        repo_dir : str
            Path to the clone of the AcidAmine_Descriptor_Predict repository.
        models_dir : str
            Path to the "trained_models" directory extracted from 3D.zip (Figshare).
        statistics : List[str]
            The ensemble statistics to predict ("boltz", "max", "min", "min_E").
        """
        self.utils = _import_utils(repo_dir, "3D")
        self.statistics = list(statistics)
        models_dir = os.path.expanduser(models_dir)

        paths = {}
        for stat in self.statistics:
            for prop, folder in ATOM_PROPERTIES_3D.items():
                paths[("amine", "atom", prop, stat)] = os.path.join(
                    models_dir, folder, prop, stat, "model_best.pt"
                )
            for prop in BOND_PROPERTIES:
                paths[("amine", "bond", prop, stat)] = os.path.join(
                    models_dir, "amines", prop, stat, "model_best.pt"
                )

        # Constructing a DimeNet++ model takes several seconds (its basis functions are built
        # with sympy), so only one model per property type is constructed, and the weights of
        # each descriptor are swapped in before predicting
        self.models = {}
        self.weights = {}
        for selection, path in paths.items():
            property_type = selection[1]
            if property_type not in self.models:
                self.models[property_type] = self.utils.load_model(*selection, {selection: path})
            self.weights[selection] = torch.load(path, map_location="cpu")

        self.atom_selection = {("amine", "atom", p): "N1" for p in ATOM_PROPERTIES_3D}
        self.atom_selection.update({("amine", "bond", p): ("N1", "C2") for p in BOND_PROPERTIES})

    def encode(self, mol):
        rdmol = Chem.MolFromMolBlock(mol.molblock, removeHs=False)
        if rdmol is None or rdmol.GetNumConformers() == 0:
            raise EncoderError("the molecule could not be read from the mol block with coordinates")

        # The models only use the heavy atoms; their indices must be the same with and without
        # hydrogen atoms, so the heavy atoms are moved to the front (as in the authors' molecules)
        order = [a.GetIdx() for a in rdmol.GetAtoms() if a.GetAtomicNum() != 1]
        order += [a.GetIdx() for a in rdmol.GetAtoms() if a.GetAtomicNum() == 1]
        rdmol = Chem.RenumberAtoms(rdmol, order)
        n1, c2 = _match_primary_amine(rdmol, mol.smiles)

        df = pd.DataFrame(
            {
                "mols": [rdmol],
                "mols_noHs": [Chem.RemoveHs(rdmol)],
                "N1": [n1],
                "C2": [c2],
                "mol_type": ["primary_amine"],
                "mol_index": [0],
            }
        )

        # Indices in BONAFIDE's (input) atom order
        n1_input, c2_input = order[n1], order[c2]

        outputs = {"n1_atom": [n1_input], "n1_c2_bond": [[n1_input, c2_input]]}
        for selection, weights in self.weights.items():
            _, property_type, prop, stat = selection
            model = self.models[property_type]
            model.load_state_dict(weights, strict=True)
            pred = self.utils.make_predictions(
                df,
                model,
                selection,
                self.atom_selection,
                keep_explicit_hydrogens=False,
                remove_Hs_except_functional=False,
            )
            outputs[f"{OUTPUT_NAMES_3D[prop]}_{stat.lower()}"] = [float(np.asarray(pred)[0])]

        return outputs
