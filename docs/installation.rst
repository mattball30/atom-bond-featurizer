##############
 Installation
##############

Please complete the following steps to install ``BONAFIDE`` and its :doc:`dependencies
<dependencies>`. The environment is managed with `pixi <https://pixi.sh>`_, which installs all
dependencies from conda-forge and PyPI. Only Multiwfn must be installed manually.

**************************************
 BONAFIDE and its Python dependencies
**************************************

#. Install **pixi** (if not already installed). See the `pixi installation guide
   <https://pixi.sh/latest/installation/>`_ for further options.

.. code:: shell

   $ curl -fsSL https://pixi.sh/install.sh | sh

2. Clone the **GitHub repository** and change to its root directory.

.. code:: shell

   $ git clone https://github.com/MolecularAI/atom-bond-featurizer.git
   $ cd atom-bond-featurizer

3. Install the **environment**. This installs BONAFIDE (in editable mode) together with all its
   dependencies at the exact versions recorded in the ``pixi.lock`` file.

.. code:: shell

   $ pixi install

.. note::

   The ``default`` environment provides the necessary dependencies for calculating features with
   BONAFIDE, whereas the ``dev`` environment contains additional dependencies for development
   purposes (*e.g.*, testing, documentation building, Jupyter kernel).

.. code:: shell

   $ pixi install -e dev

.. note::

   qmdesc (and with it PyTorch) is installed by default. To leave it out, use the ``no-qmdesc``
   environment instead (``pixi install -e no-qmdesc`` and ``pixi run -e no-qmdesc ...``). The
   qmdesc features cannot be calculated in this environment.

4. Run BONAFIDE within the environment, either by prefixing commands with ``pixi run`` or by
   activating the environment in the current shell with ``pixi shell``.

.. code:: shell

   $ pixi run python my_script.py
   $ pixi shell
   $ python my_script.py

*********
 ALFABET
*********

The machine learning model ``ALFABET`` requires Python 3.7 and TensorFlow and is therefore not
compatible with BONAFIDE's environment. It is installed in a **separate environment** named
``alfabet-env`` if the model is intended to be used through BONAFIDE.

.. code:: shell

   $ pixi install -e alfabet-env

BONAFIDE finds the Python interpreter of this environment through the ``BONAFIDE_ALFABET_PYTHON``
environment variable, which is set automatically in BONAFIDE's pixi environments. To use an ALFABET
installation in a different environment, set ``BONAFIDE_ALFABET_PYTHON`` to the path of that
environment's Python interpreter.

.. code:: shell

   $ export BONAFIDE_ALFABET_PYTHON=/path/to/alfabet_env/bin/python

**********
 Multiwfn
**********

``Multiwfn`` is not a Python dependency and therefore must be **installed separately**. Go through
the following steps to install the program (if not already installed). Additional information on the
installation can be found in section 2.1 of the `Multiwfn manual
<http://sobereva.com/multiwfn/download.html>`_. BONAFIDE was tested with the **2026.10.18** version
of Multiwfn (``Multiwfn_2026.10.18_bin_Linux_noGUI``), which should be installed to ensure that
the features are calculated as expected.

#. Download the ``Multiwfn_2026.10.18_bin_Linux_noGUI.zip`` file (Linux, no GUI) from this `webpage
   <http://sobereva.com/multiwfn/download.html>`_.

#. Unzip the folder and place it in an appropriate directory which can be accessed by BONAFIDE.

#. Ensure that the downloaded ``Multiwfn_noGUI`` file has **executable permission**. If not, this
   can be added by changing the current working directory to where Multiwfn was saved followed by
   executing the following command.

.. code:: shell

   $ chmod +x Multiwfn_noGUI

4. Add the following lines to your ``.bashrc`` file. Replace ``<path to parent directory>`` with the
   correct path.

.. code:: bash

   export Multiwfnpath=<path to parent directory>/Multiwfn_2026.10.18_bin_Linux_noGUI
   export PATH=$PATH:<path to parent directory>/Multiwfn_2026.10.18_bin_Linux_noGUI

***************************
 Finalization and checking
***************************

At any stage of the installation process, the ``check_bonafide_installation`` task can be executed
to inspect the **status of the installation**. This task checks if all dependencies (including
Multiwfn and the ``alfabet-env`` environment) are installed correctly and whether BONAFIDE can
access them.

.. code:: shell

   $ pixi run python check_bonafide_installation.py
   ...

Alternatively, for the ``dev`` environment.

.. code:: shell

   $ pixi run -e dev python check_bonafide_installation.py

If the output of this command ends with ``==> Hurray, all dependencies are installed correctly!``,
the installation process was completed **successfully**.
