"""Check that the AcidAmine 2D amine example reproduces the authors' test-set predictions.

Runs in the BONAFIDE environment. The atom and bond features of every primary amine in the
authors' test sets are calculated through BONAFIDE and compared to the published predictions
(2D/test_set_predictions in 2D.zip from Figshare, doi:10.6084/m9.figshare.25213742.v3).

    python validate_2d.py --python /path/to/acidamine-env/bin/python \
        --repo-dir /path/to/AcidAmine_Descriptor_Predict \
        --data-dir /path/to/extracted/2D
"""

import argparse
import logging
import os
import tempfile
from collections import defaultdict

import pandas as pd
from amine_featurizer import AcidAmineAmine2D
from rdkit import Chem

from bonafide import AtomBondFeaturizer

_STATS = {"Boltz": "boltz", "max": "max", "min": "min", "low_E": "min_e"}

# Test set file and the sites whose predictions are taken from it. The N1 models of primary
# amines are the combined amine models; the C2 and bond models are the primary amine models.
_TEST_FILES = {
    "amine/atom_primary_amines_predictions_targets_2D.csv.gz": ["C2"],
    "amine/bond_primary_amines_predictions_targets_2D.csv.gz": ["bond"],
    "combined_amine/atom_combined_amines_predictions_targets_2D.csv.gz": ["N1"],
}


def _expected_values(data_dir, n_max):
    """Collect (smiles, site, feature name, published prediction) from the test set files."""
    expected = defaultdict(list)
    for file_name, sites in _TEST_FILES.items():
        df = pd.read_csv(os.path.join(data_dir, "test_set_predictions", file_name))
        df = df[~df["Name_int"].astype(str).str.startswith("SecN")].head(n_max)
        for _, row in df.iterrows():
            for col in df.columns:
                if not col.startswith("prediction_"):
                    continue
                name = col[len("prediction_") :]
                site = "bond" if name.startswith("Sterimol") else name.split("_")[0]
                if site not in sites:
                    continue
                prop, stat = name, None
                for s in _STATS:
                    if name.endswith(f"_{s}"):
                        prop, stat = name[: -len(s) - 1], _STATS[s]
                if site != "bond":
                    prop = prop.split("_", 1)[1]
                expected[row["smiles"]].append((site, f"{prop.lower()}_{stat}", row[col]))
    return expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--repo-dir", required=True)
    parser.add_argument("--data-dir", required=True, help="the extracted '2D' directory")
    parser.add_argument("--n", type=int, default=500, help="molecules per test set file")
    args = parser.parse_args()

    expected = _expected_values(args.data_dir, args.n)
    deviations = defaultdict(list)
    failed = []

    init_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        for i, (smiles, values) in enumerate(expected.items()):
            f = AtomBondFeaturizer(log_file_name=os.path.join(tmp, f"bonafide_{i}.log"))
            logging.getLogger().setLevel(logging.CRITICAL)
            added = f.add_foundation_model(AcidAmineAmine2D)
            f.set_options(
                configs=[
                    ("acidamine.python", args.python),
                    ("acidamine.options.repo_dir", args.repo_dir),
                    (
                        "acidamine.options.models_dir",
                        os.path.join(args.data_dir, "trained_models"),
                    ),
                ]
            )
            f.read_input(input_value=smiles, namespace="amine")
            atom_idx = [i for n, i in added.items() if "-atom-" in n]
            bond_idx = [i for n, i in added.items() if "-bond-" in n]
            f.featurize_atoms(atom_indices="all", feature_indices=atom_idx)
            f.featurize_bonds(bond_indices="all", feature_indices=bond_idx)
            atom_cache = f.mol_vault.atom_feature_cache_n[0]
            bond_cache = f.mol_vault.bond_feature_cache[0]

            mol = f.mol_vault.mol_objects[0]
            n1, c2 = mol.GetSubstructMatch(Chem.MolFromSmarts("[NH2]C"))
            bond = mol.GetBondBetweenAtoms(n1, c2).GetIdx()

            for site, name, published in values:
                if site == "bond":
                    value = bond_cache.get(f"acidamine2D-bond-{name}", {}).get(bond)
                else:
                    atom = n1 if site == "N1" else c2
                    value = atom_cache.get(f"acidamine2D-atom-{name}", {}).get(atom)
                if not isinstance(value, float):
                    failed.append((smiles, site, name, value))
                    continue
                deviations[(site, name)].append(abs(value - published))
        os.chdir(init_dir)

    print(f"{len(expected)} molecules, {len(failed)} missing values")
    for item in failed[:10]:
        print("  missing:", item)
    rows = [
        {
            "site": s,
            "feature": n,
            "n": len(d),
            "max_abs_dev": max(d),
            "mean_abs_dev": sum(d) / len(d),
        }
        for (s, n), d in sorted(deviations.items())
    ]
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
