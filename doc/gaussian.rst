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

.. math:: e^{-i\hat{H}t/\hbar}\a e^{i\hat{H}t/\hbar} = A\a + B\ad + \mathbf{w}

where :math:`AB^T=BA^T` and :math:`AA^\dagger = BB^\dagger+\I`.


Time propagation
----------------

Once the matrix of quadratic coefficients :math:`A` and vector of linear coefficients :math:`\mathbf{d}` have been found for the Gaussian Hamiltonian, it can be shown that the Hamiltonian can always be written in the following form, up to a local phase factor :cite:`serafini2017`:

.. math:: \hat{H} = \frac{1}{2}(\r-\mathbf{d}')A(\r-\mathbf{d}')

where :math:`\mathbf{d}'=-A^{-1}\mathbf{d}` (this quantity can always be calculated, since :math:`A` is symmetric positive definite). Let's consider the case of zero linear coefficients, and non-zero linear coefficients, separately.

Zero linear coefficients
^^^^^^^^^^^^^^^^^^^^^^^^

In the Heisenberg picture, with :math:`\hat{H}=\frac{1}{2}\r A\r`, the time-evolution of the quadrature operators must satisfy the Heisenberg equations of motion:

.. math:: \frac{d}{dt}\r_j = \frac{i}{\hbar}[\hat{H},\r_j] ~~\Leftrightarrow ~~ \frac{d}{dt}\r = \Omega A \r ,

where

.. math::  \Omega = \begin{bmatrix} 0 & \I_N \\-\I_N & 0 \\\end{bmatrix}

is the `symplectic matrix <https://en.wikipedia.org/wiki/Symplectic_matrix>`_, coming from the definition of the canonical commutation relation :math:`[\r_i,\r_j]=i\hbar \Omega_{ij}`.

Solving this differential equation, we find that the symplectic Gaussian transformation describing the time-evolution of the Hamiltonian :math:`\hat{H}` acting on the quadrature operators, for time :math:`t`, is given by:

.. math:: S = \exp{\left(\Omega A t\right)}


Non-zero linear coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If, on the other hand, have non-zero linear coefficients, we need to take this into account by performing the required displacement operation. Consider the following Gaussian Hamiltonian:

.. math:: \hat{H}(\mathbf{d}') = \frac{1}{2}(\r-\mathbf{d}')A(\r-\mathbf{d}')


where :math:`\mathbf{d}'=-A^{-1}\mathbf{d}`, as before. This corresponds to the action of a `Weyl operator or displacement <https://strawberryfields.readthedocs.io/en/latest/conventions/gates.html#displacement>`_ of the form :math:`\hat{D}(\mathbf{s})` with :math:`\mathbf{s}=-\mathbf{d}'/\sqrt{2\hbar}`:

.. math::  \hat{H}(\mathbf{d}') = \frac{1}{2}\hat{D}(\mathbf{s})\r A\r \hat{D}(\mathbf{s})^\dagger = \hat{D}(\mathbf{s})\hat{H}(0)\hat{D}(\mathbf{s})^\dagger.

Calculating the time-evolution operator,

.. math:: \hat{U}(t) = e^{-i\hat{H}(d) t/\hbar} = e^{-i\hat{D}(\mathbf{s})\hat{H}(0)\hat{D}(\mathbf{s})^\dagger t} = \hat{D}(\mathbf{s})e^{-i\hat{H}(0) t}\hat{D}(\mathbf{s})^\dagger.

In order to write this as a symplectic matrix transformation, we need to move all displacement operators to the left. To do this, we can post-multiply by :math:`\I=e^{i\hat{H}(0)t}e^{-i\hat{H}(0)t}`:

.. math::
	\hat{U}(t) = \hat{D}(\mathbf{s})\left[e^{-i\hat{H}(0) t}\hat{D}(\mathbf{s})^\dagger e^{i\hat{H}(0)t}\right]e^{-i\hat{H}(0)t}

Finally, we can rewrite this as a symplectic transformation, by making the substitution :math:`e^{-i\hat{H}(0)t}\rightarrow e^{\Omega A t}` and by noting that the bracketed term is simply a displacement by :math:`-\mathbf{s}`, evolved under :math:`\hat{H}(0)` for time :math:`t`:

.. math::
	S = \hat{D}(\mathbf{s} -{e^{\Omega A t}}^T \mathbf{s}) e^{\Omega A \hbar t}


.. admonition:: Definition
	:class: defn

	For a quadratic Hamiltonian of the form :math:`\hat{H} = \frac{1}{2}\r A\r + \r^T \mathbf{d}`, the symplectic transformation :math:`S\in\mathbb{R}^{2N\times 2N}` characterizing the time-evolution unitary operator :math:`\hat{U}(t) = e^{-i\hat{H}t/\hbar}` is given by

	.. math:: S = \hat{D}(\mathbf{s} -{e^{\Omega A t}}^T \mathbf{s}) e^{\Omega A t}

	where :math:`\Omega` is the symplectic matrix, :math:`\hat{D}` the displacement operation, and :math:`\mathbf{s} = -A^{-1}\mathbf{d}/\sqrt{2\hbar}`.

.. tip::

   *Implemented in SF-OpenFermion as a quantum operation by* :class:`SFopenfermion.ops.GaussianPropagation`
