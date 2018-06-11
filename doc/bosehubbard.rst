.. role:: raw-latex(raw)
   :format: latex
   
.. role:: html(raw)
   :format: html

.. _bosehubbard:

Bose-Hubbard Hamiltonians
=========================

.. sectionauthor:: Josh Izaac <josh@xanadu.ai>

A Bose-Hubbard model describes the dynamics of multiple bosons on a lattice composed of orthonormal vertices or nodes, with the overall energy of the system dependent on the location of the bosons, surrounding potentials, and possible interactions.

Bose-Hubbard Hamiltonians have the following form:

.. math::
    \hat{H} = -J\sum_{i=1}^N\sum_{j=1}^N A_{ij} \ad_i\a_j
        + \frac{1}{2}U\sum_{i=1}^N \ad_i \a_i (\ad_i \a_i - 1)
        - \mu \sum_{i=1}^N \ad_i \a_i
        + V \sum_{i=1}^N\sum_{j=1}^N A_{ij} \ad_i \a_i \ad_j \a_j.

where:

* :math:`A` is a real symmetric matrix of ones and zeros defining the adjacency of
  each pairwise combination of nodes :math:`(i,j)` in the :math:`N`-node system.
* :math:`J` represents the transfer integral or hopping term of the boson between nodes.
* :math:`U` is the on-site interaction potential, the interaction strength between bosons on the same node. This can be attractive (:math:`U<0`) or repulsive (:math:`U>0`).
* :math:`\mu` is the chemical potential, dependent on the number of bosons per node.
* :math:`V` is the dipole-dipole or nearest neighbour interaction term, the interaction strength between bosons on adjacent nodes. Like the on-site interaction, this can be attractive or repulsive.

CV decomposition
----------------

As the Bose-Hubbard Hamiltonian is time-independent, it suffices to find a continuous-variable (CV) gate decomposition for the unitary operator :math:`\hat{U}=e^{-i\hat{H}t}` :cite:`kalajdzievski2018`. Let's consider the case :math:`V=0` (i.e. no nearest neighbour interactions). In this case, we can rewrite the Hamiltonian in the following form:


.. math::
    \hat{H} = -J\sum_{i=1}^N\sum_{j=1}^N A_{ij} \ad_i\a_j
        + \sum_{\ell=1}^N \left(\frac{1}{2}U \hat{n}_\ell^2
        - \left(\frac{1}{2}U+\mu\right) \hat{n}_\ell\right),

where :math:`\hat{n}_i=\ad_i\a_i` is the bosonic number operator. Taking the matrix exponential, and applying the Lie product formula,

.. math::
	e^{-iHt} = \lim_{k\rightarrow\infty}\left[\prod_{\substack{i,j\\i\sim j}}\exp\left({i\frac{ J t}{k}(\ad_i\a_j + \ad_j\a_i)}\right)\prod_{\ell}\exp\left(-i\frac{Ut}{2k}\hat{n}_\ell^2\right)\exp\left(i\frac{(U+2\mu)t}{2k}\hat{n}_\ell\right)\right]^k,

where :math:`i\sim j` indicates we are only summing over adjacent nodes (i.e. those where :math:`A_{ij}=1`). Truncating :math:`k` to a reasonable value, we end up with the approximation

.. math::
	e^{-iHt} = \left[\prod_{\substack{i,j\\i\sim j}}\exp\left({i\frac{ J t}{k}(\ad_i\a_j + \ad_j\a_i)}\right)\prod_{\ell}\exp\left(-i\frac{Ut}{2k}\hat{n}_\ell^2\right)\exp\left(i\frac{(U+2\mu)t}{2k}\hat{n}_\ell\right)\right]^k + \mathcal{O}(t^2/k).

Comparing these individual bracketed operators with the known CV gate set (see `here <https://strawberryfields.readthedocs.io/en/latest/conventions/gates.html>`_ for more details) we see that they correspond to a beamsplitter, Kerr gate, and phase-space rotation respectively:

* :math:`\exp\left({i\frac{ J t}{k}(\ad_i\a_j + \ad_j\a_i)}\right)\equiv BS(\theta, \phi)` where :math:`\theta=Jt/k`, :math:`\phi=\pi/2`,

* :math:`\exp\left(-i\frac{Ut}{2k}\hat{n}_\ell^2\right)\equiv K(\kappa)` where :math:`\kappa=-Ut/2k`,

* :math:`\exp\left(i\frac{(U+2\mu)t}{2k}\hat{n}_\ell\right)\equiv R(r)` where :math:`r=(U+2\mu)t/2k`.



.. admonition:: Decomposition
	:class: defn

	A Bose-Hubbard Hamiltonian with zero nearest-neighbour interaction, can be implemented to arbitrary error via a decomposition of beamsplitters, Kerr gates, and phase-space rotations.

.. tip::

   *Implemented in SFOpenBoson as a quantum operation by* :class:`sfopenboson.ops.BoseHubbardPropagation`


