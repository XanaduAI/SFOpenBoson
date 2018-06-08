SF-OpenFermion
##############

:Release: |release|
:Date: |today|

`Strawberry Fields <http://github.com/XanaduAI/strawberryfields>`_ is a full-stack Python library for
designing, simulating, and optimizing continuous variable (CV) quantum optical circuits.

`OpenFermion <http://openfermion.org>`_ is an open source package for compiling and analyzing
quantum algorithms that simulate fermionic systems.

This Strawberry Fields plugin library allows Strawberry Fields to interface with OpenFermion.

Features
========

* Construct bosonic Hamiltonians in OpenFermion, and apply the resulting time propagation to your CV quantum circuit.

* Calculates the time-evolution unitary exactly for Gaussian Hamiltonians - these can then be decomposed into the base CV gate set of Strawberry Fields using the Bloch-Messiah decomposition.

* Particular non-Gaussian gate decompositions, using the Trotter formula, are also supported, including Bose-Hubbard Hamiltonians.

* The Hamiltonians submodule contains important OpenFermion-compatible CV Hamiltonians, including those corresponding to the gate set used in Strawberry Fields.

Getting started
===============

To get the SF-OpenFermion plugin installed and running on your system, begin at the :ref:`download and installation guide <installation>`. Then, familiarise yourself with some background information on the time-propagation of :ref:`Gaussian Hamiltonians <gaussian_hamiltonians>` and :ref:`the Bose-Hubbard model <bosehubbard>`.

For getting started with using the Hafnian library in your own code, have a look at the provided :ref:`tutorials <tutorial_gaussian>`.

Finally, detailed documentation on the code and API is provided.

Support
=======

- **Source Code:** https://github.com/XanaduAI/SF-OpenFermion
- **Issue Tracker:** https://github.com/XanaduAI/SF-OpenFermion/issues

If you are having issues, please let us know, either by email or by posting the issue on our Github issue tracker.

License
=======

SF-OpenFermion is **free** and **open source**, released under the Apache License, Version 2.0.

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   installing
   research


.. toctree::
   :maxdepth: 2
   :caption: Background
   :hidden:

   gaussian
   bosehubbard
   references

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/gaussian
   tutorials/bosehubbard

.. toctree::
   :maxdepth: 2
   :caption: Code documentation
   :hidden:

   code/ops
   code/hamiltonians
   code/_bose_hubbard_trotter

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
