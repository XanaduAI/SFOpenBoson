.. role:: html(raw)
   :format: html

.. _gaussian_hamiltonians:


Gaussian Hamiltonians
======================

.. sectionauthor:: Josh Izaac <josh@xanadu.ai>

.. note:: This section assumes a familiarity with the operators and conventions used in quantum optics and continuous-variable (CV) quantum computation. For an introduction to these concepts, see the `Strawberry Fields documentation <https://strawberryfields.readthedocs.io/>`_.


Quadrature operators
--------------------

A Gaussian Hamiltonian is a Hamiltonian that can be written as a quadratic polynomial of the quadrature operators :math:`\q_i` (canonical position, also written as :math:`\x_i`) and :math:`\p_j` (canonical momentum) :cite:`weedbrook2012`:

.. math:: \hat{H} = \frac{1}{2}\r A\r + \r^T \mathbf{d}

where:

* :math:`\r=(\q_1,\dots,\q_{N},\p_1,\dots,\p_N)` is a vector of the position and momentum operators acting on qumode :math:`i`, in :math:`\q\p`-ordering,

* :math:`A\in\mathbb{R}^{2N\times 2N}` is a symmetric matrix containing the quadratic coefficients of terms of the form :math:`\hat{x}_i\hat{y}_j`,

* :math:`\mathbf{d}\in\mathbb{R}^{2N}` is a vector containing the coefficients of the linear operator terms.

For example, the following are all quadratic Hamiltonians:

* :math:`\hat{H} = \q_0^2 -q_0`
* :math:`\hat{H} = \q_0 \q_1 - \p_0\p_1`

while :math:`\hat{H}=\q_0^2\p_1`, with a third-order term in the quadrature operators, is not.


Ladder operators
--------------------

In terms of the bosonic annihilation and creation operators

.. math:: \a = \sqrt{\frac{1}{2 \hbar}} (\q +i\p), ~~~~ \ad = \sqrt{\frac{1}{2 \hbar}} (\q -i\p),

this corresponds to a Hamiltonian of the form :cite:`weedbrook2012`

.. math:: \hat{H} = i\left(\av^\dagger \mathbf{w} - \mathbf{w}\av +\av^\dagger F \av  - \av^\dagger F^\dagger \av^\dagger +\av^\dagger G {\av^\dagger}^T +{\av}^T G^\dagger \av\right)

where :math:`\av = (\a_1, \a_2,\dots,\a_N)`, :math:`\mathbf{w}\in\mathbb{C}^N` and :math:`F,G\in\mathbb{C}^{N\times N}`. This corresponds to a linear unitary `Bogoliubov transform <https://en.wikipedia.org/wiki/Bogoliubov_transformation>`_ of the annihilation and creation operators:

.. math:: e^{-iHt}\a e^{-iHt} = A\a + B\ad + \mathbf{w}

where :math:`AB^T=BA^T` and :math:`AA^\dagger = BB^\dagger+\I`.


Time propagation
----------------

Once the matrix of quadratic coefficients :math:`A` and vector of linear coefficients :math:`\mathbf{d}` have been found for the Gaussian Hamiltonian, it can be shown that the Hamiltonian can always be written in the following form, up to a constant factor :cite:`serafini2017`:

.. math:: \hat{H} = \frac{1}{2}(\r-\mathbf{d}')A(\r-\mathbf{d}')

where :math:`\mathbf{d}'=-A^{-1}\mathbf{d}` (this quantity can always be calculated, since :math:`A` is symmetric positive definite). This corresponds to the acti
on of a Weyl operator or displacement of the form :math:`\hat{D}(-\mathbf{d}'/\sqrt{2\hbar})` (see the Strawberry Fields documentation on the `displacement operator <https://strawberryfields.readthedocs.io/en/latest/conventions/gates.html#displacement>`_).

Thus, we can (for now) discount the effect of the linear coefficients, and solve the time-evolution propagation of the Hamiltonians considering only the quadratic coefficients.

In the Heisenberg picture, with :math:`\hat{H}=\frac{1}{2}\r A\r`, the time-evolution of the quadrature operators must satisfy the following differential equation:

.. math:: \frac{d}{dt}\r_j = \frac{1}{2}i[\hat{H},\r_j] ~~\Leftrightarrow ~~ \frac{d}{dt}\r = \Omega A \hbar\r ,

where

.. math::  \Omega = \begin{bmatrix} 0 & \I_N \\-\I_N & 0 \\\end{bmatrix}

is the `symplectic matrix <https://en.wikipedia.org/wiki/Symplectic_matrix>`_; this, along with :math:`\hbar`, come from the definition of the canonical commutation relation :math:`[\r_i,\r_j]=i\hbar \Omega_{ij}`.

Solving this differential equation, we find that the symplectic Gaussian transformation describing the time-evolution of the Hamiltonian :math:`\hat{H}` acting on the quadrature operators, for time :math:`t`, is given by:

.. math:: S = \exp{\left(\Omega A \hbar t\right)}

Taking into account the displacement operation due to non-zero linear coefficients, the overall symplectic transformation is given by

.. math:: S = \hat{D}\left(\frac{A^{-1}\mathbf{d}}{\sqrt{2\hbar}}\right)^\dagger\exp{\left(\Omega A \hbar t\right)}\hat{D}\left(\frac{A^{-1}\mathbf{d}}{\sqrt{2\hbar}}\right)