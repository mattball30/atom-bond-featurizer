# Foundation model example: AcidAmine 2D models (primary amines)

This example connects the 2D graph neural networks of
[AcidAmine_Descriptor_Predict](https://github.com/nsf-c-cas/AcidAmine_Descriptor_Predict)
(*Digital Discovery*, 2025, **4**, 222) to BONAFIDE as atom and bond feature generators. The models
predict conformer-ensemble DFT descriptors (Boltzmann-weighted, minimum, maximum and
lowest-energy-conformer values) from the SMILES string alone. See
[the foundation models documentation](../../../docs/foundation_models.rst) for how the pieces fit
together.

> **Disclaimer:** these models were built for **primary amines** (one `-NH2` group bound to an sp3
> carbon). Our encoder wrapper rejects any other molecule with an error. The authors also provide models
> for carboxylic acids and secondary amines, which are not part of this example.

## Files

| File | Runs in | Purpose |
|---|---|---|
| `amine_encoder.py` | AcidAmine environment | Loads the 44 GINE models and predicts the descriptors with the authors' code |
| `amine_featurizer.py` | BONAFIDE environment | Declares the model and maps its outputs onto 36 BONAFIDE features |
| `validate_2d.py` | BONAFIDE environment | Checks the predictions against the authors' published test-set predictions |

## Features

All features are only defined for the amine nitrogen (N1), its alpha carbon (C2) and the bond
between them; all other atoms and bonds are `_inaccessible`. Each descriptor comes in four ensemble
statistics: `boltz`, `max`, `min` and `min_e` (lowest-energy conformer).

- Atom features `acidamine2D-atom-<descriptor>_<statistic>`:
  - `nbo_charge` and `nmr_shift` (N1 and C2)
  - `pyrd_ar`, `nbo_lp_energy`, `nbo_lp_occupancy` and `v_bur` (N1)
- Bond features `acidamine2D-bond-<descriptor>_<statistic>` (N1-C2 bond): `sterimol_b1`,
  `sterimol_b5` and `sterimol_l`.

Molecule-level descriptors (dipole, HOMO, LUMO, ...) are out of scope, because BONAFIDE features
are atom and bond features.

## Setup

1. Clone the authors' repository (the encoder imports their prediction code; it has no licence, so
   it is not copied into BONAFIDE). The example was tested with commit `6feb0a7`.

   ```shell
   git clone https://github.com/nsf-c-cas/AcidAmine_Descriptor_Predict.git
   ```

2. Download `2D.zip` (258 MB, MD5 `313e57945f72035b4c22c0a506d0a322`) from
   [Figshare](https://doi.org/10.6084/m9.figshare.25213742.v3) and extract the trained models.
   For `validate_2d.py`, also extract the test-set predictions.

   ```shell
   cd /path/to/ckpt/dir # Choose whatever you like here

    curl -L -o 2D.zip https://ndownloader.figshare.com/files/44548739
    md5sum 2D.zip   # should be 313e57945f72035b4c22c0a506d0a322

    # Extract only the primary-amine models (plus test-set predictions, for validate_2d.py)
    unzip -q 2D.zip '2D/trained_models/amines/*' '2D/trained_models/combined_amines/*' \
        '2D/test_set_predictions/amine/*' '2D/test_set_predictions/combined_amine/*' -d tmp_extract

    mv tmp_extract/2D/trained_models/amines tmp_extract/2D/trained_models/combined_amines .
    mv tmp_extract/2D/test_set_predictions .     # optional, only needed for validate_2d.py
    rm -rf tmp_extract 2D.zip
   ```

3. Create the model environment. The following conda-forge environment works.
   Add it to the pixi manifest of your project (shown for `pixi.toml`; prefix the
   section names with `tool.pixi.` if using a `pyproject.toml`) and run `pixi install -e acidamine-env`.

   ```toml
   [feature.acidamine.dependencies]
   python = "3.10.*"
   pytorch = "1.13.*"
   pytorch_geometric = "2.4.*"
   pytorch_scatter = ">=2.1.1,<3"
   pytorch_sparse = ">=0.6.17,<0.7"
   pytorch_cluster = ">=1.6.1,<2"
   rdkit = "*"
   pandas = "*"
   numpy = "<2"
   tqdm = "*"

   [environments]
   acidamine-env = { features = ["acidamine"], no-default-feature = true }

   [activation.env]
   BONAFIDE_ACIDAMINE_PYTHON = "$PIXI_PROJECT_ROOT/.pixi/envs/acidamine-env/bin/python"
   ```

   Without the `activation.env` entry, set the interpreter with
   `f.set_options(configs=("acidamine.python", "/path/to/acidamine-env/bin/python"))`.

## Usage

In the BONAFIDE environment, from this directory:

```python
from bonafide import AtomBondFeaturizer
from amine_featurizer import AcidAmineAmine2D

f = AtomBondFeaturizer()
added = f.add_foundation_model(AcidAmineAmine2D)
f.set_options(
    configs=[
        ("acidamine.options.repo_dir", "/path/to/AcidAmine_Descriptor_Predict"),
        ("acidamine.options.models_dir", "/path/to/2D/trained_models"),
    ]
)

f.read_input(input_value="NCC1CCCCC1", namespace="amine")
f.featurize_atoms(
    atom_indices="all", feature_indices=[i for n, i in added.items() if "-atom-" in n]
)
f.featurize_bonds(
    bond_indices="all", feature_indices=[i for n, i in added.items() if "-bond-" in n]
)
print(f.return_atom_features())
print(f.return_bond_features())
```

The models are loaded once (about 2 s), and each molecule takes about 0.3 s on a CPU.

## Validation

`validate_2d.py` runs all primary amines of the authors' test sets through BONAFIDE and compares
the features with the published predictions:

```shell
python validate_2d.py --python /path/to/acidamine-env/bin/python \
    --repo-dir /path/to/AcidAmine_Descriptor_Predict --data-dir /path/to/2D
```

For all 500 primary amines of the test set, all 44 descriptor and statistic combinations are
reproduced to within float32 precision: the largest absolute deviation is 6.6e-5 (N1 NMR shift,
values of about 230 ppm), and below 1e-6 for all charges, lone pair properties and pyramidalization
values.
