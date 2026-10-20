# Foundation model example: AcidAmine models (primary amines)

This example connects the graph neural networks of
[AcidAmine_Descriptor_Predict](https://github.com/nsf-c-cas/AcidAmine_Descriptor_Predict)
(*Digital Discovery*, 2025, **4**, 222) to BONAFIDE as atom and bond feature generators. The models
predict conformer-ensemble DFT descriptors (Boltzmann-weighted, minimum, maximum and
lowest-energy-conformer values):

- the **2D models** (GINE) from the SMILES string alone (`input_mode = "2D"`);
- the **3D models** (DimeNet++) from one 3D conformer at a time (`input_mode = "3D"`). See
[the foundation models documentation](../../../docs/foundation_models.rst) for how the pieces fit
together.

> **Disclaimer:** these models were built for **primary amines** (one `-NH2` group bound to an sp3
> carbon). Our encoder wrapper rejects any other molecule with an error. The authors also provide models
> for carboxylic acids and secondary amines, which are not part of this example.

## Files

| File | Runs in | Purpose |
|---|---|---|
| `amine_encoder.py` | AcidAmine environment | `AmineEncoder2D` loads the 44 GINE models, `AmineEncoder3D` the 28 DimeNet++ models (one conformer at a time); both predict with the authors' code |
| `amine_featurizer.py` | BONAFIDE environment | Declares both models (`AcidAmineAmine2D`, `AcidAmineAmine3D`) and maps their outputs onto 36 and 28 BONAFIDE features |
| `validate.py` | BONAFIDE environment | Compare the predictions with the authors' published test-set predictions |

## Features

All features are only defined for the amine nitrogen (N1), its alpha carbon (C2) and the bond
between them; all other atoms and bonds are `_inaccessible`. Each descriptor comes in four ensemble
statistics: `boltz`, `max`, `min` and `min_e` (lowest-energy conformer).

- Atom features `acidamine2D-atom-<descriptor>_<statistic>`:
  - `nbo_charge` and `nmr_shift` (N1 and C2)
  - `pyrd_ar`, `nbo_lp_energy`, `nbo_lp_occupancy` and `v_bur` (N1)
- Bond features `acidamine2D-bond-<descriptor>_<statistic>` (N1-C2 bond): `sterimol_b1`,
  `sterimol_b5` and `sterimol_l`.
- 3D atom features `acidamine_dimenet3D-atom-<descriptor>_<statistic>` (N1): `pyrd_ar`,
  `nbo_lp_energy`, `nbo_lp_occupancy` and `v_bur` (the authors call the pyramidalization
  `pyr_agranat` in their 3D models and `N1_pyrd_Ar` in their 2D models).
- 3D bond features `acidamine_dimenet3D-bond-<descriptor>_<statistic>` (N1-C2 bond): `sterimol_b1`,
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
   For `validate.py`, also extract the test-set predictions.

   ```shell
   cd /path/to/ckpt/dir # Choose whatever you like here

    curl -L -o 2D.zip https://ndownloader.figshare.com/files/44548739
    md5sum 2D.zip   # should be 313e57945f72035b4c22c0a506d0a322

    # Extract only the primary-amine models (plus test-set predictions, for validate.py)
    unzip -q 2D.zip '2D/trained_models/amines/*' '2D/trained_models/combined_amines/*' \
        '2D/test_set_predictions/amine/*' '2D/test_set_predictions/combined_amine/*' -d tmp_extract

    mv tmp_extract/2D/trained_models/amines tmp_extract/2D/trained_models/combined_amines .
    mv tmp_extract/2D/test_set_predictions .     # optional, only needed for validate.py
    rm -rf tmp_extract 2D.zip
   ```

   For the 3D models, do the same with `3D.zip` (521 MB, MD5
   `6a1d1289a30e2fcdba5532d96dbc38f7`, `https://ndownloader.figshare.com/files/44550749`). For
   `validate.py`, also extract the test-set predictions and the authors' conformers:

   ```shell
   unzip -q 3D.zip '3D/trained_models/amines/*' '3D/trained_models/combined_amines/*' \
       '3D/test_set_predictions/amine_*' '3D/data/3D_model_primaryamine_rdkit_conformers.csv'
   ```

3. Create the model environment (the same environment runs the 2D and 3D models). The following conda-forge environment works.
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

### 3D models

The 3D models need a conformer ensemble, e.g., from an SD file. BONAFIDE sends each conformer to
the model, and the predictions are averaged with `reduce=True`:

```python
from amine_featurizer import AcidAmineAmine3D

added3d = f.add_foundation_model(AcidAmineAmine3D)
f.set_options(
    configs=[
        ("acidamine_dimenet.options.repo_dir", "/path/to/AcidAmine_Descriptor_Predict"),
        ("acidamine_dimenet.options.models_dir", "/path/to/3D/trained_models"),
    ]
)

f.read_input(input_value="conformers.sdf", namespace="amine", input_format="file")
f.featurize_atoms(
    atom_indices="all", feature_indices=[i for n, i in added3d.items() if "-atom-" in n]
)
print(f.return_atom_features(reduce=True))
```

Loading takes about 25 s (building the DimeNet++ basis functions), and each conformer about 0.5 s.

#### Differences from the paper

The authors generate an RDKit conformer ensemble for each molecule (ETKDG, MMFF94 optimization,
Butina clustering to at most 20 conformers), predict each conformer and report the **arithmetic
mean**. Here, the predictions are made for **the conformers read into BONAFIDE**, whatever their
origin (RDKit, CREST, DFT, ...), and averaged by BONAFIDE. The results therefore match the paper
only for the same conformers; with other ensembles, they differ by how much the model's prediction
changes between conformers (see the validation below). Note that the 3D models were trained on
MMFF94 conformers, so very different geometries (e.g., from DFT) are outside their training
distribution.

## Validation

### 2D models

`validate.py 2d` runs all primary amines of the authors' test sets through BONAFIDE and compares
the features with the published predictions:

```shell
python validate.py 2d --python /path/to/acidamine-env/bin/python \
    --repo-dir /path/to/AcidAmine_Descriptor_Predict --data-dir /path/to/2D
```

`--data-dir` must contain `test_set_predictions`; use `--models-dir` if the models are not in
`<data-dir>/trained_models`.

For all 500 primary amines of the test set, all 44 descriptor and statistic combinations are
reproduced to within float32 precision: the largest absolute deviation is 6.6e-5 (N1 NMR shift,
values of about 230 ppm), and below 1e-6 for all charges, lone pair properties and pyramidalization
values.

### 3D models

`validate.py 3d` compares the averaged 3D features with the published test-set predictions in two
ways:

```shell
python validate.py 3d --python /path/to/acidamine-env/bin/python \
    --repo-dir /path/to/AcidAmine_Descriptor_Predict --data-dir /path/to/3D --n-a 1000 --n-b 100
```

- **A. The authors' conformers:** TODO_A
- **B. New conformers** generated with the authors' protocol (different random ensembles): TODO_B
