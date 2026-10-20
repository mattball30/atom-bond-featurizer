###################
 Foundation models
###################

BONAFIDE can use **machine learning (foundation) models as atom and bond feature generators**, also
when they cannot be installed in the BONAFIDE environment. The model runs in its **own Python
environment**, and BONAFIDE communicates with it through a worker process. Two small classes
connect a model to BONAFIDE:

-  An **encoder** (runs in the model environment): it loads the model and returns the model outputs
   for a molecule.
-  A **featurizer** (runs in the BONAFIDE environment): it defines where the model lives, whether it
   takes SMILES strings or 3D structures, and how its outputs map onto BONAFIDE atom and bond
   features.

BONAFIDE does not come with any foundation model, however, a worked example for the 2D models of
`AcidAmine_Descriptor_Predict <https://github.com/nsf-c-cas/AcidAmine_Descriptor_Predict>`_ is
available in ``examples/foundation_models/acidamine``.

*************
 The encoder
*************

The encoder is a class inheriting from ``FoundationModelEncoder`` that implements ``encode()`` and,
optionally, ``load()``. BONAFIDE does **not** need to be installed in the model environment: the
worker makes the base class importable as ``bonafide_encoder`` (it depends on the Python standard
library only).

.. code:: python

   # my_encoder.py (runs in the model environment)
   from bonafide_encoder import EncoderError, FoundationModelEncoder


   class MyEncoder(FoundationModelEncoder):
       def load(self, checkpoint, device="cpu"):
           # Called once when the worker starts; the arguments are the "options" of the featurizer
           self.model = load_my_model(checkpoint, device)

       def encode(self, mol):
           # Called once per molecule (2D) or conformer (3D)
           if mol.charge != 0:
               raise EncoderError("only neutral molecules are supported")
           pred = self.model.predict(mol.smiles)
           return {
               "charges": pred.atom_charges,  # one value per atom
               "embedding": pred.atom_embeddings,  # one vector per atom
               "bond_orders": pred.bond_orders,  # one value per bond ...
               "bond_atoms": pred.bond_atom_pairs,  # ... identified by its atom pair
           }

``encode()`` receives a ``MolInput`` object with the following attributes:

-  ``smiles``: the canonical SMILES string (without hydrogen atoms).
-  ``molblock``: an MDL mol block of the molecule in BONAFIDE's atom order, with coordinates for 3D
   input.
-  ``atomic_numbers``, ``bonds`` (begin atom, end atom, bond order), and ``coordinates`` (3D input
   only), all in BONAFIDE's atom and bond order.
-  ``charge``, ``multiplicity`` (``None`` if not set), ``has_explicit_hydrogens``, and
   ``input_mode``.

It returns a dictionary of named outputs: lists, numpy arrays, or torch tensors. Raising
``EncoderError`` reports a handled error for the molecule (e.g., a molecule outside of the model's
domain); any other exception is reported with its traceback. Everything the model prints is
redirected to the worker's stderr, which is included in BONAFIDE's error messages.

****************
 The featurizer
****************

The featurizer is a class inheriting from ``FoundationModelFeaturizer``. It is configured with
class attributes; no methods need to be implemented.

.. code:: python

   from bonafide import AtomBondFeaturizer
   from bonafide.foundation import FoundationModelFeaturizer, Output


   class MyModel(FoundationModelFeaturizer):
       origin = "mymodel"
       input_mode = "2D"
       encoder = "/path/to/my_encoder.py:MyEncoder"
       options = {"checkpoint": "/path/to/weights.pt", "device": "cpu"}
       outputs = [
           Output("charges", "atom", name="partial_charge"),
           Output("embedding", "atom", store="vector"),
           Output("bond_orders", "bond", indices_key="bond_atoms"),
       ]


   f = AtomBondFeaturizer()
   added = f.add_foundation_model(MyModel)
   # {'mymodel2D-atom-partial_charge': 590, 'mymodel2D-atom-embedding': 591,
   #  'mymodel2D-bond-bond_orders': 592}

   f.read_input(input_value="NCC(=O)O", namespace="glycine")
   f.featurize_atoms(atom_indices="all", feature_indices=[590, 591])
   f.featurize_bonds(bond_indices="all", feature_indices=592)

The class attributes are:

-  ``origin``: the name of the model. It is used in the feature names
   (``{origin}{2D|3D}-{atom|bond}-{name}``) and as configuration section.
-  ``input_mode``: ``"2D"`` or ``"3D"``.

   -  ``"2D"``: the encoder receives the molecule without coordinates and is called once per
      molecule (the result is reused for all conformers). Use it for models that take SMILES
      strings or molecular graphs, including models that generate 3D structures internally. The
      features are 2D features.
   -  ``"3D"``: the encoder additionally receives the coordinates and is called once per conformer.
      The features are 3D features.

-  ``encoder``: the encoder file and class, ``"path/to/file.py:ClassName"``.
-  ``outputs``: the mapping of the encoder outputs onto features (see below).
-  ``python`` (optional): the Python interpreter of the model environment (see below).
-  ``options`` (optional): a dictionary passed to the encoder's ``load()`` method.
-  ``timeout`` (optional, default 600): the time in seconds after which loading the model or
   encoding a molecule is aborted.
-  ``requires_charge`` and ``requires_multiplicity`` (optional, default ``False``): whether the
   charge or multiplicity must be set before the features can be calculated.

``add_foundation_model()`` must be called in every Python session (like ``add_custom_featurizer()``)
and returns the names and indices of the added features. Because these indices depend on what was
added before, refer to the features by name in scripts.

Mapping outputs onto features
=============================

Each ``Output`` turns one encoder output into one feature.

-  ``key``: the name of the encoder output.
-  ``feature_type``: ``"atom"`` or ``"bond"``.
-  ``name``: the feature name (default: ``key``).
-  ``store``: ``"scalar"`` (default; one number per atom or bond) or ``"vector"`` (one list of
   numbers per atom or bond, stored as comma-separated string, e.g., embeddings).
-  ``index``: store only this component of a vector-valued output as scalar feature.
-  ``data_type``: ``"float"`` (default), ``"int"``, or ``"bool"`` for scalar features.
-  ``atom_order``: which atom order the output refers to.

   -  ``"canonical"`` (default for 2D): the atom order of ``Chem.MolFromSmiles(mol.smiles)``,
      followed by ``Chem.AddHs()`` if ``mol.has_explicit_hydrogens`` is ``True``. Do not
      re-canonicalize the SMILES string in the encoder.
   -  ``"input"`` (default for 3D): BONAFIDE's atom order (as in ``mol.molblock`` and
      ``mol.coordinates``).

   BONAFIDE maps the values onto its own atom and bond indices.

-  ``indices_key``: the name of an encoder output that identifies the atoms (atom indices) or bonds
   (pairs of atom indices) the values belong to. Without it, an output must contain one value for
   every atom or bond in order. Use it for models that only predict values for some atoms or bonds;
   all other atoms or bonds are set to ``"_inaccessible"``. Identifying bonds by atom pairs is
   recommended, as bond orders differ between toolkits.
-  ``nan``: how missing values (NaN or ``None``) are treated, ``"missing"`` (default; the feature is
   ``"_inaccessible"`` for that atom or bond) or ``"error"`` (the feature calculation fails).

Values for atoms that do not exist in the molecule read into BONAFIDE are dropped. For example, if
the molecule was read in without explicit hydrogen atoms, hydrogen atom predictions are not part of
the output. Add hydrogen atoms to the input to obtain them.

If the ``Output`` declarations cannot express the mapping, override the ``map_outputs()`` method of
the featurizer class.

*************************************
 Setting up the model environment
*************************************

BONAFIDE looks for the Python interpreter of the model environment in this order:

#. the ``python`` setting, which can be changed at runtime:

   .. code:: python

      f.set_options(configs=("mymodel.python", "/path/to/model_env/bin/python"))

#. the ``BONAFIDE_<ORIGIN>_PYTHON`` environment variable, e.g., ``BONAFIDE_MYMODEL_PYTHON``.

With pixi, the model environment can be added to a project's manifest as separate environment
(``no-default-feature = true`` keeps BONAFIDE's dependencies out of it), and the environment variable
can be set automatically. For a ``pixi.toml`` file (prefix all section names with ``tool.pixi.`` in a
``pyproject.toml`` file):

.. code:: toml

   [feature.mymodel.dependencies]
   python = "3.10.*"
   pytorch = "*"

   [environments]
   mymodel-env = { features = ["mymodel"], no-default-feature = true }

   [activation.env]
   BONAFIDE_MYMODEL_PYTHON = "$PIXI_PROJECT_ROOT/.pixi/envs/mymodel-env/bin/python"

Install it with ``pixi install -e mymodel-env``.

The other settings can be changed with ``set_options()`` as well, e.g.,
``f.set_options(configs=("mymodel.options.device", "cuda"))``.

********************
 The worker process
********************

-  The worker is started when the first feature of a model is calculated and loads the model once.
   It is shared by all ``AtomBondFeaturizer`` instances in the Python session and stopped when the
   session ends, or earlier with ``close_foundation_models()`` (e.g., to free GPU memory).
-  Changing the interpreter or the options starts a new worker.
-  Results are kept in memory per molecule input, so a model is not called again for the same
   molecule (2D) or conformer (3D).
-  If the worker crashes or exceeds the timeout, the features of that molecule fail with an error
   message (including the worker's stderr), and the worker is restarted on the next request.
