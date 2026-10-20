"""A toy "foundation model" encoder used in foundation_model_example.ipynb.

It needs RDKit, so it can run with the Python interpreter of the BONAFIDE environment. A real
encoder looks the same, but loads and runs a model in its own environment.
"""

from bonafide_encoder import EncoderError, FoundationModelEncoder
from rdkit import Chem
from rdkit.Chem import AllChem

_SUPPORTED_ELEMENTS = {"H", "C", "N", "O", "F", "S", "Cl", "Br"}


class ToyEncoder(FoundationModelEncoder):
    def load(self, embedding_scale=1.0):
        # A real encoder would load the model weights here (once per worker)
        self.embedding_scale = embedding_scale

    def encode(self, mol):
        # Parse the SMILES string without re-canonicalizing it: the returned atom indices then
        # refer to BONAFIDE's "canonical" atom order
        rdmol = Chem.MolFromSmiles(mol.smiles)
        if mol.has_explicit_hydrogens:
            rdmol = Chem.AddHs(rdmol)

        unsupported = {a.GetSymbol() for a in rdmol.GetAtoms()} - _SUPPORTED_ELEMENTS
        if unsupported:
            # A handled error: reported for this molecule, without traceback
            raise EncoderError(f"elements {sorted(unsupported)} are not supported by the toy model")

        AllChem.ComputeGasteigerCharges(rdmol)
        atoms = list(rdmol.GetAtoms())
        heteroatoms = [a for a in atoms if a.GetSymbol() not in ["C", "H"]]

        return {
            # One value per atom
            "charge": [a.GetDoubleProp("_GasteigerCharge") for a in atoms],
            # One vector per atom (a pretend embedding)
            "embedding": [
                [
                    self.embedding_scale * a.GetAtomicNum(),
                    float(a.GetDegree()),
                    float(a.GetIsAromatic()),
                    float(a.GetTotalNumHs()),
                ]
                for a in atoms
            ],
            # Values for some atoms only, identified by their indices
            "lone_pairs": [
                (Chem.GetPeriodicTable().GetNOuterElecs(a.GetAtomicNum()) - a.GetTotalValence()) / 2
                for a in heteroatoms
            ],
            "lone_pair_atoms": [a.GetIdx() for a in heteroatoms],
            # One value per bond, identified by its atom pair
            "bond_order": [b.GetBondTypeAsDouble() for b in rdmol.GetBonds()],
            "bond_atoms": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in rdmol.GetBonds()],
        }
