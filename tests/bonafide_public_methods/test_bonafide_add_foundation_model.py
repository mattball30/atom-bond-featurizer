"""Test functions for the ``add_foundation_model()`` method in the ``bonafide.bonafide`` module
and the foundation model infrastructure it uses.

The dummy encoder (``tests/data/foundation_dummy_encoder.py``) runs in a worker process started
with the interpreter that runs the tests.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List

import pytest
from rdkit import Chem

from bonafide.utils.base_foundation_model import FoundationModelFeaturizer, Output
from bonafide.utils import foundation_model_worker as worker_module
from bonafide.utils.foundation_model_worker import close_all_workers

if TYPE_CHECKING:
    from bonafide.bonafide import AtomBondFeaturizer

_ENCODER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "foundation_dummy_encoder.py"
)


class Dummy2D(FoundationModelFeaturizer):
    origin = "dummy"
    input_mode = "2D"
    encoder = f"{_ENCODER}:DummyEncoder"
    python = sys.executable
    timeout = 60.0
    outputs = [
        Output("atomic_num", "atom"),
        Output("embedding", "atom", store="vector"),
        Output("embedding", "atom", name="degree", index=1),
        Output("with_nan", "atom"),
        Output("with_nan", "atom", name="with_nan_error", nan="error"),
        Output("site_value", "atom", indices_key="site_indices"),
        Output("bond_order", "bond", indices_key="bond_pairs"),
        Output("bond_order", "bond", name="bond_order_in_order"),
    ]


class Dummy3D(FoundationModelFeaturizer):
    origin = "dummy3"
    input_mode = "3D"
    encoder = f"{_ENCODER}:DummyEncoder"
    python = sys.executable
    timeout = 60.0
    outputs = [
        Output("input_atomic_num", "atom"),
        Output("x_coord", "atom"),
    ]


@pytest.fixture
def in_tmp_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Run the test in a temporary directory and stop all workers afterwards."""
    monkeypatch.chdir(tmp_path)
    yield
    close_all_workers()


def _features(
    f: AtomBondFeaturizer, added: Dict[str, int], feature_type: str, names: List[str]
) -> Any:
    indices = [added[f"{f_name}"] for f_name in names]
    if feature_type == "atom":
        f.featurize_atoms(atom_indices="all", feature_indices=indices)
        return f.return_atom_features()
    f.featurize_bonds(bond_indices="all", feature_indices=indices)
    return f.return_bond_features()


@pytest.mark.add_foundation_model
def test_add_foundation_model(
    fresh_featurizer: Callable[..., AtomBondFeaturizer], in_tmp_dir: None
) -> None:
    """Test for the ``add_foundation_model()`` method: registration."""
    f = fresh_featurizer()
    n_features = len(f._feature_info)
    added = f.add_foundation_model(Dummy2D)

    assert list(added.values()) == list(range(n_features, n_features + len(Dummy2D.outputs)))
    assert "dummy2D-atom-atomic_num" in added
    assert "dummy2D-bond-bond_order" in added
    assert f._feature_info[added["dummy2D-atom-embedding"]]["data_type"] == "str"
    assert f._feature_info[added["dummy2D-atom-atomic_num"]]["data_type"] == "float"
    assert f._feature_info[added["dummy2D-atom-atomic_num"]]["dimensionality"] == "2D"
    assert f._feature_config["dummy"]["python"] == sys.executable
    assert f._feature_config["dummy"]["encoder"] == f"{_ENCODER}:DummyEncoder"
    assert set(added.values()) <= set(f.list_atom_features().index) | set(
        f.list_bond_features().index
    )

    # The origin can only be used once
    with pytest.raises(ValueError, match="already in use"):
        f.add_foundation_model(Dummy2D)


@pytest.mark.add_foundation_model
@pytest.mark.parametrize("smiles", ["NCC(=O)O", "OC(=O)CN", "[H]OC(=O)C([H])([H])N([H])[H]"])
def test_foundation_model_2d_atom_mapping(
    fresh_featurizer: Callable[..., AtomBondFeaturizer], in_tmp_dir: None, smiles: str
) -> None:
    """Test the mapping of canonical-order outputs onto the input atom order."""
    f = fresh_featurizer()
    added = f.add_foundation_model(Dummy2D)
    f.read_input(input_value=smiles, namespace="gly")
    df = _features(
        f,
        added,
        "atom",
        [
            "dummy2D-atom-atomic_num",
            "dummy2D-atom-embedding",
            "dummy2D-atom-degree",
            "dummy2D-atom-with_nan",
            "dummy2D-atom-site_value",
        ],
    )

    mol = f.mol_vault.mol_objects[0]
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        z = atom.GetAtomicNum()
        row = df.loc[idx]
        assert row["dummy2D-atom-atomic_num"] == float(z)
        assert row["dummy2D-atom-embedding"] == f"{float(z)},{float(atom.GetDegree())},0.5"
        assert row["dummy2D-atom-degree"] == float(atom.GetDegree())
        assert row["dummy2D-atom-with_nan"] == ("_inaccessible" if z == 6 else 1.0)
        assert row["dummy2D-atom-site_value"] == (7.0 if z == 7 else "_inaccessible")


@pytest.mark.add_foundation_model
def test_foundation_model_2d_bond_mapping(
    fresh_featurizer: Callable[..., AtomBondFeaturizer], in_tmp_dir: None
) -> None:
    """Test the mapping of bond outputs (atom pairs and canonical bond order)."""
    f = fresh_featurizer()
    added = f.add_foundation_model(Dummy2D)
    f.read_input(input_value="OC(=O)C=CN", namespace="mol")
    df = _features(
        f, added, "bond", ["dummy2D-bond-bond_order", "dummy2D-bond-bond_order_in_order"]
    )

    mol = f.mol_vault.mol_objects[0]
    for bond in mol.GetBonds():
        expected = bond.GetBondTypeAsDouble()
        assert df.loc[bond.GetIdx(), "dummy2D-bond-bond_order"] == expected
        assert df.loc[bond.GetIdx(), "dummy2D-bond-bond_order_in_order"] == expected


@pytest.mark.add_foundation_model
def test_foundation_model_nan_error(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: Callable[..., AtomBondFeaturizer],
    in_tmp_dir: None,
) -> None:
    """Test that NaN values are reported as error if requested."""
    f = fresh_featurizer()
    added = f.add_foundation_model(Dummy2D)
    f.read_input(input_value="NCC", namespace="mol")
    f.featurize_atoms(atom_indices=0, feature_indices=added["dummy2D-atom-with_nan_error"])

    assert "dummy2D-atom-with_nan_error" not in f.mol_vault.atom_feature_cache_n[0]
    assert any("contains missing values" in r.msg for r in caplog.records if r.levelname == "ERROR")


@pytest.mark.add_foundation_model
def test_foundation_model_encoder_error(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: Callable[..., AtomBondFeaturizer],
    in_tmp_dir: None,
) -> None:
    """Test that handled encoder errors are reported per molecule."""
    f = fresh_featurizer()
    added = f.add_foundation_model(Dummy2D)
    f.read_input(input_value="ClCCN", namespace="mol")
    f.featurize_atoms(atom_indices=0, feature_indices=added["dummy2D-atom-atomic_num"])

    errors = [r.msg for r in caplog.records if r.levelname == "ERROR"]
    assert any("chlorine is outside of the model's domain" in e for e in errors)
    assert not any("Traceback" in e for e in errors)


@pytest.mark.add_foundation_model
def test_foundation_model_no_python(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: Callable[..., AtomBondFeaturizer],
    in_tmp_dir: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the interpreter lookup through set_options and the environment variable."""
    f = fresh_featurizer()
    added = f.add_foundation_model(Dummy2D)
    f.set_options(configs=("dummy.python", None))
    f.read_input(input_value="NCC", namespace="mol")
    feature_idx = added["dummy2D-atom-atomic_num"]

    monkeypatch.delenv("BONAFIDE_DUMMY_PYTHON", raising=False)
    f.featurize_atoms(atom_indices=0, feature_indices=feature_idx)
    errors = [r.msg for r in caplog.records if r.levelname == "ERROR"]
    assert any("BONAFIDE_DUMMY_PYTHON" in e for e in errors)

    monkeypatch.setenv("BONAFIDE_DUMMY_PYTHON", sys.executable)
    f.featurize_atoms(atom_indices=0, feature_indices=feature_idx)
    assert f.mol_vault.atom_feature_cache_n[0]["dummy2D-atom-atomic_num"][0] == 7.0


@pytest.mark.add_foundation_model
def test_foundation_model_options_and_memo(
    fresh_featurizer: Callable[..., AtomBondFeaturizer],
    in_tmp_dir: None,
    fetch_data_file: Callable[[str], str],
) -> None:
    """Test that 2D input is encoded once per ensemble and that options reach the encoder."""
    f = fresh_featurizer()
    added = f.add_foundation_model(Dummy2D)
    f.read_input(
        input_value=fetch_data_file("spiro_mol.sdf"), namespace="spiro", input_format="file"
    )
    assert len(f.mol_vault.mol_objects) == 5

    f.set_options(configs=("dummy.options", {"scale": 2.0}))
    _features(f, added, "atom", ["dummy2D-atom-atomic_num"])
    for conf_idx, mol in enumerate(f.mol_vault.mol_objects):
        cache = f.mol_vault.atom_feature_cache_n[conf_idx]["dummy2D-atom-atomic_num"]
        for atom in mol.GetAtoms():
            assert cache[atom.GetIdx()] == 2.0 * atom.GetAtomicNum()

    workers = list(worker_module._WORKERS.values())
    assert len(workers) == 1
    assert len(workers[0]._memo) == 1


@pytest.mark.add_foundation_model
def test_foundation_model_3d(
    fresh_featurizer: Callable[..., AtomBondFeaturizer],
    in_tmp_dir: None,
    fetch_data_file: Callable[[str], str],
) -> None:
    """Test that 3D models receive the coordinates of each conformer in the input order."""
    f = fresh_featurizer()
    added = f.add_foundation_model(Dummy3D)
    f.read_input(
        input_value=fetch_data_file("spiro_mol.sdf"), namespace="spiro", input_format="file"
    )
    f.featurize_atoms(
        atom_indices="all",
        feature_indices=[added["dummy33D-atom-input_atomic_num"], added["dummy33D-atom-x_coord"]],
    )

    for conf_idx, mol in enumerate(f.mol_vault.mol_objects):
        positions = mol.GetConformer(0).GetPositions()
        cache = f.mol_vault.atom_feature_cache_n[conf_idx]
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            assert cache["dummy33D-atom-input_atomic_num"][idx] == float(atom.GetAtomicNum())
            assert cache["dummy33D-atom-x_coord"][idx] == pytest.approx(positions[idx][0])

    # One encoder call per conformer
    workers = list(worker_module._WORKERS.values())
    assert len(workers[0]._memo) == 5


@pytest.mark.add_foundation_model
@pytest.mark.parametrize(
    "options, message",
    [
        ({"fail_load": True}, "dummy load failure"),
        ({"crash": True}, "terminated unexpectedly"),
        ({"sleep": 5.0}, "did not respond within"),
    ],
)
def test_foundation_model_worker_failures(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: Callable[..., AtomBondFeaturizer],
    in_tmp_dir: None,
    options: Dict[str, Any],
    message: str,
) -> None:
    """Test that load failures, crashes, and timeouts are reported and recovered from."""
    f = fresh_featurizer()
    added = f.add_foundation_model(Dummy2D)
    f.set_options(configs=[("dummy.options", options), ("dummy.timeout", 1.0)])
    f.read_input(input_value="NCC", namespace="mol")
    feature_idx = added["dummy2D-atom-atomic_num"]

    f.featurize_atoms(atom_indices=0, feature_indices=feature_idx)
    errors = [r.msg for r in caplog.records if r.levelname == "ERROR"]
    assert any(message in e for e in errors)
    assert "dummy2D-atom-atomic_num" not in f.mol_vault.atom_feature_cache_n[0]

    # A new worker with working options succeeds
    f.set_options(configs=[("dummy.options", {}), ("dummy.timeout", 60.0)])
    f.featurize_atoms(atom_indices=0, feature_indices=feature_idx)
    assert f.mol_vault.atom_feature_cache_n[0]["dummy2D-atom-atomic_num"][0] == 7.0

    f.close_foundation_models()
    assert worker_module._WORKERS == {}


@pytest.mark.add_foundation_model
def test_foundation_model_shape_mismatch(
    caplog: pytest.LogCaptureFixture,
    fresh_featurizer: Callable[..., AtomBondFeaturizer],
    in_tmp_dir: None,
) -> None:
    """Test that outputs with the wrong number of entries are reported as error."""

    class DummySparse(Dummy2D):
        origin = "dummy_sparse"
        outputs = [Output("site_value", "atom")]  # sparse output without indices_key

    f = fresh_featurizer()
    added = f.add_foundation_model(DummySparse)
    f.read_input(input_value="NCC", namespace="mol")
    f.featurize_atoms(atom_indices=0, feature_indices=added["dummy_sparse2D-atom-site_value"])

    errors = [r.msg for r in caplog.records if r.levelname == "ERROR"]
    assert any("use indices_key" in e for e in errors)


def _modified(**attrs: Any) -> type:
    return type("Modified", (Dummy2D,), attrs)


@pytest.mark.add_foundation_model
@pytest.mark.parametrize(
    "model, error",
    [
        (str, TypeError),
        (Chem.Mol, TypeError),
        (_modified(origin="1abc"), ValueError),
        (_modified(origin="qmdesc"), ValueError),
        (_modified(input_mode="4D"), ValueError),
        (_modified(encoder="no_colon.py"), ValueError),
        (_modified(encoder="missing_file.py:DummyEncoder"), ValueError),
        (_modified(python="/does/not/exist/python"), ValueError),
        (_modified(options={"x": object()}), TypeError),
        (_modified(timeout=0), ValueError),
        (_modified(outputs=[]), ValueError),
        (_modified(outputs=[Output("a", "atom"), Output("b", "atom", name="a")]), ValueError),
    ],
)
def test_add_foundation_model_invalid(
    fresh_featurizer: Callable[..., AtomBondFeaturizer],
    in_tmp_dir: None,
    model: Any,
    error: Any,
) -> None:
    """Test for the ``add_foundation_model()`` method: invalid model classes."""
    f = fresh_featurizer()
    n_features = len(f._feature_info)
    with pytest.raises(error):
        f.add_foundation_model(model)
    assert len(f._feature_info) == n_features


@pytest.mark.add_foundation_model
@pytest.mark.parametrize(
    "kwargs",
    [
        {"feature_type": "molecule"},
        {"feature_type": "atom", "store": "matrix"},
        {"feature_type": "atom", "store": "vector", "data_type": "float"},
        {"feature_type": "atom", "store": "vector", "index": 0},
        {"feature_type": "atom", "atom_order": "random"},
        {"feature_type": "atom", "nan": "ignore"},
        {"feature_type": "atom", "name": "has space"},
    ],
)
def test_output_invalid(kwargs: Dict[str, Any]) -> None:
    """Test the validation of ``Output`` objects."""
    with pytest.raises(ValueError):
        Output("key", **kwargs)
