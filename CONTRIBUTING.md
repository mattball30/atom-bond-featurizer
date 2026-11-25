# How to contribute

Contributions to the projects are very welcome either through

-   using the tool and reporting on its usage
-   opening issues or pull requests
-   suggesting any kinds of extensions/improvements
-   by getting in direct contact via email (lukas.sigmund@astrazeneca.com)

## Environment

For development work, BONAFIDE should be installed in its development environment, which is
managed with [pixi](https://pixi.sh). Multiwfn (version 2026.10.18) must be installed separately as
described in the
[installation guide](https://molecularai.github.io/atom-bond-featurizer/installation.html).

```shell
pixi install -e dev
pixi install -e alfabet-env
pixi run -e dev python check_bonafide_installation.py
```

The dependencies are defined in `pyproject.toml` and locked in `pixi.lock`. After changing
dependencies, update the lock file with `pixi lock` and commit both files.

The `dev` environment includes `ipykernel`, so it can be selected as the kernel for the notebooks in
`examples/` (the `.pixi/envs/dev/bin/python` interpreter).

## Code style and formatting

The codebase complies with [mypy](https://mypy.readthedocs.io/en/stable/)'s static type checking as
specified in `mypy.ini`. Type checking is executed as follows.

```shell
pixi run -e dev mypy
```

[Ruff](https://docs.astral.sh/ruff/formatter/) is used for Python code formatting and is
configured as specified in `ruff.toml`.

```shell
pixi run -e dev format-check
pixi run -e dev format
```

[Prettier](https://prettier.io/) was used for markdown formatting.

For docstring coverage analysis, [interrogate](https://interrogate.readthedocs.io/en/latest/) is
used.

```shell
pixi run -e dev docstring-coverage
```

## Documentation

Every function, class, or method should have a docstring with a full description of its
functionality along with its parameters / attributes. The
[numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html) standard is followed. The
web documentation is built using [sphinx](https://www.sphinx-doc.org/en/master/) in the `docs`
folder making use of the [furo](https://github.com/pradyunsg/furo) theme. The HTML output is
written to `docs/_build/html` (the `BUILDDIR` variable in `docs/Makefile`).

```shell
pixi run -e dev docs
```

[rstfmt](https://github.com/dzhu/rstfmt) is used for reStructuredText file formatting.

```shell
pixi run -e dev rstfmt <file path> -w 100
```

## Testing

Tests are written using [pytest](https://docs.pytest.org/en/stable/). New functionality introduced
to the package should be covered by suitable tests. Global fixtures useful for test implementation
can be found in `tests/conftest.py`.

For running all tests, execute either of the following commands from the root directory of the
repository:

```shell
pixi run -e dev test
```

```shell
pixi run -e dev test --recalc_all
```

The additional `--recalc_all` option forces all tests to be re-executed even if their results were
already cached as successful. This includes the explicit recalculation of all available features for
at least one example. This takes around 30 min.

Test coverage can also be reported.

```shell
pixi run -e dev test-coverage
```

It is also possible to run a slightly reduced test suite as it is done in CI/CD. It excludes for
example Multiwfn as a required dependency for running the tests.

```shell
pixi run -e dev test-quick
```
