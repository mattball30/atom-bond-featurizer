"""Compare the AcidAmine amine example with the authors' published test-set predictions.

Runs in the BONAFIDE environment. The published predictions are in 2D.zip and 3D.zip from Figshare
(doi:10.6084/m9.figshare.25213742.v3).

2D models: the atom and bond features of every primary amine in the authors' test sets are
calculated through BONAFIDE and compared to the published predictions
(2D/test_set_predictions).

    python validate.py 2d --python /path/to/acidamine-env/bin/python \
        --repo-dir /path/to/AcidAmine_Descriptor_Predict \
        --data-dir /path/to/extracted/2D

3D models: the features are calculated through BONAFIDE for each conformer, averaged over the
ensemble, and compared with the published predictions (3D/test_set_predictions) in two ways:

A. With the authors' own conformers (3D/data/3D_model_primaryamine_rdkit_conformers.csv). This
   checks that BONAFIDE reproduces the published predictions.
B. With new conformer ensembles generated with the authors' protocol (RDKit ETKDG + MMFF94,
   Butina clustering to at most 20 conformers). This shows how much the predictions change when
   the conformers come from elsewhere, compared with the error of the model itself (the
   deviation of the published predictions from the DFT values).

    python validate.py 3d --python /path/to/acidamine-env/bin/python \
        --repo-dir /path/to/AcidAmine_Descriptor_Predict \
        --data-dir /path/to/extracted/3D --n-a 100 --n-b 50
"""

import argparse
import glob
import logging
import os
import tempfile
from collections import defaultdict

import pandas as pd
from amine_featurizer import AcidAmineAmine2D, AcidAmineAmine3D
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.ML.Cluster import Butina

from bonafide import AtomBondFeaturizer

_STATS_2D = {"Boltz": "boltz", "max": "max", "min": "min", "low_E": "min_e"}
_STATS_3D = {"Boltz": "boltz", "max": "max", "min": "min", "lowE": "min_e"}

# Test set file and the sites whose predictions are taken from it. The N1 models of primary
# amines are the combined amine models; the C2 and bond models are the primary amine models.
_TEST_FILES_2D = {
    "amine/atom_primary_amines_predictions_targets_2D.csv.gz": ["C2"],
    "amine/bond_primary_amines_predictions_targets_2D.csv.gz": ["bond"],
    "combined_amine/atom_combined_amines_predictions_targets_2D.csv.gz": ["N1"],
}

_NAMES_3D = {
    "NBO_LP_energy": "nbo_lp_energy",
    "NBO_LP_occupancy": "nbo_lp_occupancy",
    "pyr_agranat": "pyrd_ar",
    "Vbur": "v_bur",
    "Sterimol_B1": "sterimol_b1",
    "Sterimol_B5": "sterimol_b5",
    "Sterimol_L": "sterimol_l",
}


def _expected_values_2d(data_dir, n_max):
    """Collect (smiles, site, feature name, published prediction) from the 2D test set files."""
    expected = defaultdict(list)
    for file_name, sites in _TEST_FILES_2D.items():
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
                for s in _STATS_2D:
                    if name.endswith(f"_{s}"):
                        prop, stat = name[: -len(s) - 1], _STATS_2D[s]
                if site != "bond":
                    prop = prop.split("_", 1)[1]
                expected[row["smiles"]].append((site, f"{prop.lower()}_{stat}", row[col]))
    return expected


def _published_3d(data_dir):
    """Read the published 3D predictions and DFT values: {feature: DataFrame(Name_int, pred, dft)}."""
    published = {}
    for path in glob.glob(os.path.join(data_dir, "test_set_predictions", "amine_*.csv")):
        _, kind, rest = os.path.basename(path)[:-4].split("_", 2)
        if kind == "mol":
            continue
        prop, stat = rest.rsplit("_", 1)
        df = pd.read_csv(path).iloc[:, [0, 2, 3]]
        df.columns = ["Name_int", "dft", "pred"]
        published[f"acidamine_dimenet3D-{kind}-{_NAMES_3D[prop]}_{_STATS_3D[stat]}"] = df.set_index(
            "Name_int"
        )
    return published


def _generate_ensemble(smiles, n_confs=20, seed=42):
    """Conformer ensemble with the authors' protocol (utils.generate_conformer_dataframe)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    cids = list(
        AllChem.EmbedMultipleConfs(
            mol, numConfs=n_confs * 5, pruneRmsThresh=0.25, maxAttempts=50, randomSeed=seed
        )
    )
    for cid in cids:
        AllChem.MMFFOptimizeMolecule(mol, confId=cid)
    heavy = Chem.RemoveHs(mol)
    dists = [
        rdMolAlign.GetBestRMS(heavy, heavy, cids[i], cids[j])
        for i in range(len(cids))
        for j in range(i)
    ]
    thresh = 0.2
    clusters = Butina.ClusterData(dists, len(cids), thresh, isDistData=True, reordering=True)
    while len(clusters) > n_confs:
        thresh += 0.1
        clusters = Butina.ClusterData(dists, len(cids), thresh, isDistData=True, reordering=True)
    return [Chem.Mol(mol, confId=cids[c[0]]) for c in clusters]


def _featurize_ensemble(f, features, mols, sdf_path):
    """Calculate the 3D features for a conformer ensemble and average them over the conformers."""
    writer = Chem.SDWriter(sdf_path)
    for i, m in enumerate(mols):
        m.SetProp("_Name", f"conf_{i}")
        writer.write(m)
    writer.close()

    f.read_input(input_value=sdf_path, namespace="amine", input_format="file")
    mol = f.mol_vault.mol_objects[0]
    n1, c2 = mol.GetSubstructMatch(Chem.MolFromSmarts("[NH2]C"))
    bond = mol.GetBondBetweenAtoms(n1, c2).GetIdx()
    atom_idx = [i for n, i in features.items() if "-atom-" in n]
    bond_idx = [i for n, i in features.items() if "-bond-" in n]
    f.featurize_atoms(atom_indices=n1, feature_indices=atom_idx)
    f.featurize_bonds(bond_indices=bond, feature_indices=bond_idx)

    means = {}
    for name in features:
        if "-atom-" in name:
            caches, idx = f.mol_vault.atom_feature_cache_n, n1
        else:
            caches, idx = f.mol_vault.bond_feature_cache, bond
        values = [c.get(name, {}).get(idx) for c in caches]
        if all(isinstance(v, float) for v in values):
            means[name] = sum(values) / len(values)
    return means


def _new_featurizer(model, origin, args, log_file_name):
    """Create a featurizer with the foundation model added and configured."""
    f = AtomBondFeaturizer(log_file_name=log_file_name)
    logging.getLogger().setLevel(logging.CRITICAL)
    added = f.add_foundation_model(model)
    f.set_options(
        configs=[
            (f"{origin}.python", args.python),
            (f"{origin}.options.repo_dir", args.repo_dir),
            (f"{origin}.options.models_dir", args.models_dir),
        ]
    )
    return f, added


def validate_2d(args):
    expected = _expected_values_2d(args.data_dir, args.n)
    deviations = defaultdict(list)
    failed = []

    init_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        for i, (smiles, values) in enumerate(expected.items()):
            f, added = _new_featurizer(
                AcidAmineAmine2D, "acidamine", args, os.path.join(tmp, f"bonafide_{i}.log")
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


def validate_3d(args):
    published = _published_3d(args.data_dir)
    ids = sorted(set.intersection(*[set(df.index) for df in published.values()]))
    conformers = pd.read_csv(
        os.path.join(args.data_dir, "data", "3D_model_primaryamine_rdkit_conformers.csv")
    )
    conformers = conformers[conformers.Name_int.isin(ids)]

    init_dir = os.getcwd()
    results = {"A": defaultdict(dict), "B": defaultdict(dict)}
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        f, features = _new_featurizer(
            AcidAmineAmine3D, "acidamine_dimenet", args, os.path.join(tmp, "bonafide.log")
        )
        for comparison, n in [("A", args.n_a), ("B", args.n_b)]:
            for i, name_int in enumerate(ids[:n]):
                blocks = conformers[conformers.Name_int == name_int].mol_block
                mols = [Chem.MolFromMolBlock(b, removeHs=False) for b in blocks]
                if comparison == "B":
                    mols = _generate_ensemble(Chem.MolToSmiles(Chem.RemoveHs(mols[0])))
                means = _featurize_ensemble(
                    f, features, mols, os.path.join(tmp, f"{comparison}_{i}.sdf")
                )
                for feature, value in means.items():
                    results[comparison][feature][name_int] = value
                print(f"{comparison}: {i + 1}/{n} ({len(mols)} conformers)", flush=True)
        f.close_foundation_models()
        os.chdir(init_dir)

    rows = []
    for feature, df in sorted(published.items()):
        row = {"feature": feature.split("-", 1)[1]}
        a = pd.Series(results["A"][feature], dtype=float)
        b = pd.Series(results["B"][feature], dtype=float)
        row["n_A"] = len(a)
        row["A_vs_published_max"] = (a - df.pred[a.index]).abs().max()
        row["n_B"] = len(b)
        row["B_vs_published_mae"] = (b - df.pred[b.index]).abs().mean()
        row["B_vs_dft_mae"] = (b - df.dft[b.index]).abs().mean()
        row["published_vs_dft_mae"] = (df.pred[b.index] - df.dft[b.index]).abs().mean()
        rows.append(row)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.3g}".format)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="models", required=True)
    parser_2d = subparsers.add_parser("2d", help="validate the 2D models")
    parser_2d.add_argument("--n", type=int, default=500, help="molecules per test set file")
    parser_3d = subparsers.add_parser("3d", help="validate the 3D models")
    parser_3d.add_argument("--n-a", type=int, default=100, help="molecules for comparison A")
    parser_3d.add_argument("--n-b", type=int, default=50, help="molecules for comparison B")
    for p in (parser_2d, parser_3d):
        p.add_argument("--python", required=True)
        p.add_argument("--repo-dir", required=True)
        p.add_argument(
            "--data-dir",
            required=True,
            help="the extracted 2D or 3D directory containing 'test_set_predictions' (and "
            "'data' for 3D)",
        )
        p.add_argument(
            "--models-dir",
            help="directory containing the 'amines' and 'combined_amines' model folders "
            "(default: <data-dir>/trained_models)",
        )
    args = parser.parse_args()
    args.models_dir = args.models_dir or os.path.join(args.data_dir, "trained_models")

    if args.models == "2d":
        validate_2d(args)
    else:
        validate_3d(args)


if __name__ == "__main__":
    main()
