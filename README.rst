SFOpenBoson
==============

This Strawberry Fields plugin library allows Strawberry Fields to interface with OpenFermion.

`Strawberry Fields <http://github.com/XanaduAI/strawberryfields>`_ is a full-stack Python library for
designing, simulating, and optimizing continuous variable (CV) quantum optical circuits.

`OpenFermion <http://openfermion.org>`_ is an open source package for compiling and analyzing
quantum algorithms that simulate fermionic systems.


Dependencies
-------------

SFOpenBoson depends on the following Python packages:

* `Python <http://python.org/>`_ >=3.5
* `NumPy <http://numpy.org/>`_  >=1.13.3
* `SciPy <http://scipy.org/>`_  >=1.0.0
* `Strawberry Fields <http://strawberryfields.readthedocs.io/>`_ >=0.7.2
* `OpenFermion <https://github.com/quantumlib/OpenFermion>`_ >=0.6

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


Code authors
------------

Josh Izaac.

If you are doing research using Strawberry Fields, please cite `our whitepaper <https://arxiv.org/abs/1804.03159>`_:

  Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. *arXiv*, 2018. arXiv:1804.03159


Support
-------

- **Source Code:** https://github.com/XanaduAI/SFOpenBoson
- **Issue Tracker:** https://github.com/XanaduAI/SFOpenBoson/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

We also have a `Strawberry Fields Slack channel <https://u.strawberryfields.ai/slack>`_ -
come join the discussion and chat with our Strawberry Fields team.


License
-------

SFOpenBoson is **free** and **open source**, released under the Apache License, Version 2.0.
