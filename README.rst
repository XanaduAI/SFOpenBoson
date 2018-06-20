SFOpenBoson
###########

.. image:: https://img.shields.io/travis/XanaduAI/SFOpenBoson/master.svg?style=for-the-badge
    :alt: Travis
    :target: https://travis-ci.org/XanaduAI/SFOpenBoson

.. image:: https://img.shields.io/codecov/c/github/xanaduai/SFOpenBoson/master.svg?style=for-the-badge
    :alt: Codecov coverage
    :target: https://codecov.io/gh/XanaduAI/SFOpenBoson

.. image:: https://img.shields.io/codacy/grade/4a3ad81b88d149e791a02ee3f924eb4f.svg?style=for-the-badge
    :alt: Codacy grade
    :target: https://app.codacy.com/app/XanaduAI/SFOpenBoson?utm_source=github.com&utm_medium=referral&utm_content=XanaduAI/SFOpenBoson&utm_campaign=badger

.. image:: https://img.shields.io/readthedocs/SFOpenBoson.svg?style=for-the-badge
    :alt: Read the Docs
    :target: https://sfopenboson.readthedocs.io

.. image:: https://img.shields.io/pypi/pyversions/SFOpenBoson.svg?style=for-the-badge
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/SFOpenBoson


This Strawberry Fields plugin library allows Strawberry Fields to interface with OpenFermion.

`Strawberry Fields <http://github.com/XanaduAI/strawberryfields>`_ is a full-stack Python library for
designing, simulating, and optimizing continuous variable (CV) quantum optical circuits.

`OpenFermion <http://openfermion.org>`_ is an open source package for compiling and analyzing
quantum algorithms that simulate fermionic systems.


Features
========

* Construct bosonic Hamiltonians in OpenFermion, and apply the resulting time propagation using  a CV quantum circuit.

* Calculates the time-evolution unitary exactly for Gaussian Hamiltonians – these can then be decomposed into the base CV gate set of Strawberry Fields using the Bloch-Messiah decomposition.

* Particular non-Gaussian gate decompositions, using the Trotter formula, are also supported, including Bose-Hubbard Hamiltonians.

* The Hamiltonians submodule contains important OpenFermion-compatible CV Hamiltonians, including those corresponding to the gate set used in Strawberry Fields.

To get started, please see the online `documentation <https://sfopenboson.readthedocs.io>`_


Installation
============

Installation of SFOpenBoson, as well as all required Python packages mentioned above, can be done using pip:
::

    $ python -m pip install sfopenboson


Code authors
============

Josh Izaac.

If you are doing research using Strawberry Fields, please cite `our whitepaper <https://arxiv.org/abs/1804.03159>`_:

  Nathan Killoran, Josh Izaac, Nicolás Quesada, Ville Bergholm, Matthew Amy, and Christian Weedbrook. Strawberry Fields: A Software Platform for Photonic Quantum Computing. *arXiv*, 2018. arXiv:1804.03159


Support
=======

- **Source Code:** https://github.com/XanaduAI/SFOpenBoson
- **Issue Tracker:** https://github.com/XanaduAI/SFOpenBoson/issues

If you are having issues, please let us know by posting the issue on our Github issue tracker.

We also have a `Strawberry Fields Slack channel <https://u.strawberryfields.ai/slack>`_ -
come join the discussion and chat with our Strawberry Fields team.


License
=======

SFOpenBoson is **free** and **open source**, released under the Apache License, Version 2.0.
