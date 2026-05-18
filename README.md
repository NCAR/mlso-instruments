# mlso-instruments

This package contains Python utilities for analyzing and displaying data from
instruments at the Mauna Loa Solar Observatory (MLSO)

[![Read the Docs](https://app.readthedocs.org/projects/mlso-instruments/badge/?version=latest)](https://mlso-instruments.readthedocs.io/en/latest/)
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/NCAR/mlso-instruments/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/NCAR/mlso-instruments/tree/main)
[![Version](https://img.shields.io/pypi/v/mlso-instruments.svg)](https://pypi.org/project/mlso-instruments/)
![Status](https://img.shields.io/pypi/status/mlso-instruments.svg)

## Installation

### Installing from PyPI

The easiest way to install the MLSO API client is via the released versions on
PyPI. This is the recommended method for most users.

```console
pip install mlso-instruments
```

If you want to upgrade an existing installation, do:

```console
pip install -U mlso-instruments
```

### Installing from source

The source code can be found on the [repo's GitHub page]. Use git or download
a ZIP file with contents of the source.

[repo's GitHub page]: https://github.com/NCAR/mlso-instruments

Once you have the source code, install the Python portion of the package:

```console
cd mlso-instruments
pip install .
```

If you intend to make changes to the code, install the dev requirements and
allow changes to the code to automatically be used:

```console
pip install -e .[dev]
```

For IDL, simply put the `idl/` directory in your `IDL_PATH`.


## Usage

See the [documentation] for help on using the package, including the API
Endpoints, the bindings for Python and IDL, and the command-line interface.

[documentation]: https://mlso-instruments.readthedocs.io/en/latest/index.html
