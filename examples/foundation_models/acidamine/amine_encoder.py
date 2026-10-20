"""Encoder for the 2D primary amine models of AcidAmine_Descriptor_Predict.

This file runs in the AcidAmine environment (not in the BONAFIDE environment). It wraps the GINE
models from https://github.com/nsf-c-cas/AcidAmine_Descriptor_Predict (Digital Discovery, 2025,
4, 222), which predict conformer-ensemble DFT descriptors (Boltzmann-weighted, minimum, maximum,
and lowest-energy-conformer values) of specific atoms and bonds of primary amines from SMILES.

The models were trained for primary amines only (one -NH2 group bound to an sp3 carbon). Other
molecules are rejected with an EncoderError.
"""

import os
import sys

import pandas as pd
from bonafide_encoder import EncoderError, FoundationModelEncoder
from rdkit import Chem

# Atom properties: site of the functional group -> model folder (as in the authors' notebook)
ATOM_PROPERTIES = {
    "N1_NBO_charge": "combined_amines",
    "N1_NMR_shift": "combined_amines",
    "N1_pyrd_Ar": "combined_amines",
    "N1_NBO_LP_energy": "combined_amines",
    "N1_NBO_LP_occupancy": "combined_amines",
    "N1_V_bur": "combined_amines",
    "C2_NBO_charge": "amines",
    "C2_NMR_shift": "amines",
}

# Bond properties of the N1-C2 bond
BOND_PROPERTIES = ["Sterimol_B1", "Sterimol_B5", "Sterimol_L"]

STATISTICS = ["boltz", "max", "min", "min_E"]

# Primary amine bound to an sp3 carbon that is not part of an amide-like group
_AMINE_SMARTS = Chem.MolFromSmarts("[NX3;H2;!$(N[#6,#7,#16]=[O,S,N])][CX4]")


class AmineEncoder(FoundationModelEncoder):
    def load(self, repo_dir, models_dir, statistics=STATISTICS):
        """Load all models.

        Parameters
        ----------
        repo_dir : str
            Path to the clone of the AcidAmine_Descriptor_Predict repository.
        models_dir : str
            Path to the "trained_models" directory extracted from 2D.zip (Figshare).
        statistics : List[str]
            The ensemble statistics to predict ("boltz", "max", "min", "min_E").
        """
        sys.path.insert(0, os.path.join(os.path.expanduser(repo_dir), "2D"))
        import utils  # the authors' prediction code

        self.utils = utils
        self.statistics = list(statistics)
        models_dir = os.path.expanduser(models_dir)

        self.models = {}
        for stat in self.statistics:
            for prop, folder in ATOM_PROPERTIES.items():
                selection = ("pamine", "atom", prop, stat)
                path = os.path.join(models_dir, folder, prop, stat, "model_best.pt")
                self.models[selection] = utils.load_model(*selection, {selection: path})
            for prop in BOND_PROPERTIES:
                selection = ("pamine", "bond", prop, stat)
                path = os.path.join(models_dir, "amines", prop, stat, "model_best.pt")
                self.models[selection] = utils.load_model(*selection, {selection: path})

        self.atom_selection = {("pamine", "atom", p): p.split("_")[0] for p in ATOM_PROPERTIES}
        self.atom_selection.update({("pamine", "bond", p): ("N1", "C2") for p in BOND_PROPERTIES})

    def encode(self, mol):
        # Do not re-canonicalize: the atom indices returned must refer to the atom order of
        # MolFromSmiles(mol.smiles) followed by AddHs (the "canonical" order in BONAFIDE)
        mol_h = Chem.AddHs(Chem.MolFromSmiles(mol.smiles))

        matches = mol_h.GetSubstructMatches(_AMINE_SMARTS)
        if len(matches) != 1 or len(mol_h.GetSubstructMatches(Chem.MolFromSmarts("[NH2]C"))) != 1:
            raise EncoderError(
                "the AcidAmine amine models require exactly one primary amine (-NH2 bound to an "
                f"sp3 carbon) but found {len(matches)} in '{mol.smiles}'"
            )
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
            }
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
