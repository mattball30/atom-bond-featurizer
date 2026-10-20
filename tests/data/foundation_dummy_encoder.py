"""Dummy foundation model encoder for testing (runs with the test interpreter).

The outputs are derived from the molecule so that the index mapping can be verified: the
"atomic_num" output of every atom must end up on the atom with that atomic number.
"""

import os
import time

from bonafide_encoder import EncoderError, FoundationModelEncoder
from rdkit import Chem


class DummyEncoder(FoundationModelEncoder):
    def load(self, fail_load=False, crash=False, sleep=0.0, scale=1.0):
        if fail_load:
            raise RuntimeError("dummy load failure")
        self.crash = crash
        self.sleep = sleep
        self.scale = scale

    def encode(self, mol):
        if self.crash:
            os._exit(1)
        time.sleep(self.sleep)
        print("this goes to stderr, not to the protocol")
        if "Cl" in mol.smiles:
            raise EncoderError("chlorine is outside of the model's domain")

        # Canonical atom order, as documented in MolInput
        can = Chem.MolFromSmiles(mol.smiles)
        if mol.has_explicit_hydrogens:
            can = Chem.AddHs(can)

        atomic_num = [float(a.GetAtomicNum()) * self.scale for a in can.GetAtoms()]
        out = {
            "atomic_num": atomic_num,
            "embedding": [[a.GetAtomicNum(), a.GetDegree(), 0.5] for a in can.GetAtoms()],
            "with_nan": [float("nan") if a.GetAtomicNum() == 6 else 1.0 for a in can.GetAtoms()],
            "site_value": [7.0 for a in can.GetAtoms() if a.GetAtomicNum() == 7],
            "site_indices": [a.GetIdx() for a in can.GetAtoms() if a.GetAtomicNum() == 7],
            "bond_order": [b.GetBondTypeAsDouble() for b in can.GetBonds()],
            "bond_pairs": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in can.GetBonds()],
            "input_atomic_num": [float(z) for z in mol.atomic_numbers],
            "x_coord": [c[0] for c in mol.coordinates] if mol.coordinates is not None else [],
        }
        return out
