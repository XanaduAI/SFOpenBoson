.. _installation:

Installation and Downloads
##########################


Dependencies
-------------

SFOpenBoson depends on the following Python packages:

* `Strawberry Fields <http://strawberryfields.readthedocs.io/>`_ >=0.7.3
* `OpenFermion <https://github.com/quantumlib/OpenFermion>`_ >=0.7

These can be installed using pip, or, if on linux, using your package manager (i.e., ``apt`` if on a Debian-based system.)


Installation
------------

Installation of SFOpenBoson, as well as all required Python packages mentioned above, can be done using pip:
::

    $ python -m pip install sfopenboson


Software tests
--------------

To ensure that the SFOpenBoson plugin is working correctly after installation, the test suite can be run by navigating to the source code folder and running: ::

	$ make test


Documentation
-------------

The SFOpenBoson documentation is built automatically and hosted at `Read the Docs <https://sfopenboson.readthedocs.io>`_.

To build the documentation locally, the following additional packages are required:

* `Sphinx <http://sphinx-doc.org/>`_ >=1.5
* `sphinxcontrib-bibtex <https://sphinxcontrib-bibtex.readthedocs.io/en/latest/>`_ >=0.3.6

These can be installed via ``pip``: ::

    $ python -m pip install sphinx sphinxcontrib-bibtex --user

To build the HTML documentation, go to the top-level directory and run the command
::

  $ make doc

The documentation can then be found in the ``docs/_build/html/`` directory.

