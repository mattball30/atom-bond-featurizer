# How to contribute

Contributions to the projects are very welcome either through

-   using the tool and reporting on its usage
-   opening issues or pull requests
-   suggesting any kinds of extensions/improvements
-   by getting in direct contact via email (lukas.sigmund@astrazeneca.com)

## Environment

For development work, BONAFIDE should be installed in its development environment, which is
managed with [pixi](https://pixi.sh).

```shell
pixi install -e dev
pixi run -e dev check
```

The dependencies are defined in `pyproject.toml` and locked in `pixi.lock`. After changing
dependencies, update the lock file with `pixi lock` and commit both files.

The following tasks are available in the `dev` environment:

```shell
pixi run -e dev test         # full test suite (requires Multiwfn and the alfabet environment)
pixi run -e dev test-quick   # test suite as run in the CI
pixi run -e dev mypy         # static type checking
pixi run -e dev format-check # ruff format check
pixi run -e dev docs         # build the documentation
```

## Code style and formatting

The codebase complies with [mypy](https://mypy.readthedocs.io/en/stable/)'s static type checking as
specified in `mypy.ini`. Type checking is executed as follows.

```shell
mypy src/ > mypy.out
```

[Ruff](https://docs.astral.sh/ruff/formatter/) was used for Python code formatting and was
configured as specified in `ruff.toml`.

[Prettier](https://prettier.io/) was used for markdown formatting.

For docstring coverage analysis, [interrogate](https://interrogate.readthedocs.io/en/latest/) was
used.

```shell
interrogate -vv bonafide -i -I > interrogate.out
```

## Documentation

Every function, class, or method should have a docstring with a full description of its
functionality along with its parameters / attributes. The
[numpy docstring](https://numpydoc.readthedocs.io/en/latest/format.html) standard is followed. The
web documentation is built using [sphinx](https://www.sphinx-doc.org/en/master/) in the `docs`
folder making use of the [furo](https://github.com/pradyunsg/furo) theme.

**IMPORTANT**: adjust the `BUILDDIR` variable in the `docs/Makefile` file before building the
documentation.

```shell
make html
```

[rstfmt](https://github.com/dzhu/rstfmt) was used for reStructuredText file formatting

```shell
rstfmt <file path> -w 100
```

## Testing

Tests are written using [pytest](https://docs.pytest.org/en/stable/). New functionality introduced
to the package should be covered by suitable tests. Global fixtures useful for test implementation
can be found in `tests/conftest.py`.

For running all tests, execute either of the following commands from the root directory of the
repository:

```shell
pytest -s tests/
```

```shell
pytest -s tests/ --recalc_all
```

The additional `--recalc_all` option forces all tests to be re-executed even if their results were
already cached as successful. This includes the explicit recalculation of all available features for
at least one example. This takes around 30 min.

Test coverage can also be reported.

```shell
pytest -s --cov-report=term-missing --cov=bonafide tests/ > pytest-coverage.out
```
