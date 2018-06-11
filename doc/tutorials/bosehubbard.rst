.. _tutorial_BH:

Bose-Hubbard time propagation
==============================

.. sectionauthor:: Josh Izaac <josh@xanadu.ai>

In this tutorial, I will walk through an example of Hamiltonian simulation of a Bose-Hubbard model, using Strawberry Fields and OpenFermion.

On a lattice
------------

OpenFermion provides a convenient Hamiltonian function to automatically generate Bose-Hubbard Hamiltonians on a two-dimension lattice. For example, to generate a Bose-Hubbard Hamiltonian on a size :math:`1\times 2` lattice, with on-site and nearest neighbor interactions,

>>> from openfermion.hamiltonians import bose_hubbard
>>> bose_hubbard(x_dimension=1, y_dimension=2, tunneling=1, interaction=2,
... chemical_potential=0., dipole=3., periodic=False)
1.0 [0^ 0 0^ 0] +
-1.0 [0^ 0] +
3.0 [0^ 0 1^ 1] +
-1.0 [0^ 1] +
-1.0 [0 1^] +
1.0 [1^ 1 1^ 1] +
-1.0 [1^ 1]

For more information regarding this function, please see the `OpenFermion documentation <http://openfermion.readthedocs.io/en/latest/openfermion.html#openfermion.hamiltonians.bose_hubbard>`_).

Let's use this capability, along with the Hamiltonian propagation and decomposition tools of the SF-OpenFermion plugin, to perform Bose-Hubbard simulations in Strawberry Fields. Consider the `Hamiltonian simulation <https://strawberryfields.readthedocs.io/en/latest/algorithms/hamiltonian_simulation.html>`_ algorithm in the Strawberry Fields documentation; to reproduce these results, we first generate a Bose-Hubbard Hamiltonian on a non-periodic :math:`1\times 2` lattice, with tunneling coefficient -1, and on-site interaction strength 1.5.

>>> H = bose_hubbard(1, 2, 1, 1.5)

To simulate the time-propagation of the Hamiltonian in StrawberryFields, we also need to import the :class:`~.BoseHubbardPropagation` class from the SF-OpenFermion plugin:

>>> import strawberryfields as sf
>>> from strawberryfields.ops import *
>>> from SFopenfermion.ops import BoseHubbardPropagation

:class:`~.BoseHubbardPropagation` accepts the following arguments:

* ``operator``: a Bose-Hubbard Hamiltonian, either in the form of a ``BosonOperator`` or ``QuadOperator``.

* ``t`` (float): the time propagation value. If not provided, default value is 1.

* ``k`` (int): the number of products in the truncated Lie product formula. Increasing this parameter increases the numerical accuracy of the decomposition, but also increases the depth of the circuit and the computational time.

* ``mode`` (str): By default, ``mode='local'`` and the Hamiltonian is assumed to apply to only the applied qumodes. For example, if ``QuadOperator('q0 p1') | (q[2], q[4])``, then ``q0`` acts on ``q[2]``, and ``p1`` acts on ``q[4]``.

Alternatively, you can set ``mode='global'``, and the Hamiltonian is instead applied to the entire register by directly matching qumode numbers of the defined Hamiltonian; i.e., ``q0`` is applied to ``q[0]``, ``p1`` is applied to ``q[1]``, etc.

Let's set up the two qumode quantum circuit - each mode corresponds to a node in the lattice - and propagating the Bose-Hubbard Hamiltonian ``H`` we defined in the previous section, starting from the initial state :math:`\ket{0,2}` in the Fock space, for time :math:`t=1.086` and Lie product truncation :math:`k=20`:

>>> eng, q = sf.Engine(2)
>>> with eng:
...     Fock(2) | q[1]
...     BoseHubbardPropagation(H, 1.086, 20) | q

Now, we can run this simulation using the `Fock backend <https://strawberryfields.readthedocs.io/en/latest/code/backend.fock.html>`_, and output the Fock state probabilities at time :math:`t=1.086`:

.. note:: In the Bose-Hubbard model, the number of particles in the system remains constant, so we do not need to increase the cutoff dimension of the simulation beyond the total number of photons in the initial state.

>>> state = eng.run('fock', cutoff_dim=3)
>>> state.fock_prob([2,0])
0.52240124572001967
>>> state.fock_prob([1,1])
0.23565287685672467
>>> state.fock_prob([0,2])
0.24194587742325965

We can see that this matches the results obtained in the Strawberry Fields documentation.

Note that, as before, we can output the decomposition as applied by the Strawberry Fields engine using ``eng.print_applied()``.


On an arbitrary network
-----------------------

Alternatively, we are not bound to use the ``bose_hubbard`` function from OpenFermion; we can define our own Bose-Hubbard Hamiltonian using the ``BosonOperator`` class. For example, consider a Bose-Hubbard model constrained to a 3-vertex cycle graph; that is, the graph formed by connecting three vertices to each other in a cycle.

>>> from openfermion.ops import BosonOperator

Let's define this Hamiltonian using OpenFermion. First, constructing the tunneling terms between each pair of adjacent modes:

>>> J = 1
>>> H = BosonOperator('0 1^', -J) + BosonOperator('0^ 1', -J)
>>> H += BosonOperator('0 2^', -J) + BosonOperator('0^ 2', -J)
>>> H += BosonOperator('1 2^', -J) + BosonOperator('1^ 2', -J)

Next, let's add an on-site interaction term, with strength :math:`U=1.5`:

>>> U = 1.5
>>> H += BosonOperator('0^ 0 0^ 0', 0.5*U) - BosonOperator('0^ 0', 0.5*U)
>>> H += BosonOperator('1^ 1 1^ 1', 0.5*U) - BosonOperator('1^ 1', 0.5*U)
>>> H += BosonOperator('2^ 2 2^ 2', 0.5*U) - BosonOperator('2^ 2', 0.5*U)

.. note:: If a Hamiltonian that cannot be written in the form of Bose-Hubbard model is passed to :class:`~.BoseHubbardPropagation`, a :py:class:`~.BoseHubbardError` is returned.

As before, we use :class:`~.BoseHubbardPropagation` to simulate this model for time :math:`t=1.086`, starting from initial state :math:`\ket{2,0}`. Due to the increased size of this model, let's increase the Lie product truncation to :math:`k=100`:

>>> eng, q = sf.Engine(3)
>>> with eng:
...     Fock(2) | q[0]
...     BoseHubbardPropagation(H, 1.086, 100) | q

Running the circuit, and checking some output probabilities:

>>> state = eng.run('fock', cutoff_dim=3)
>>> for i in ([2,0,0], [1,1,0], [1,0,1], [0,2,0], [0,1,1], [0,0,2]):
>>> 	print(state.fock_prob(i))
0.0854670760113
0.0492551749656
0.0487405644017
0.311517563612
0.197891000006
0.307128621004

To verify this result, we can construct the :math:`6\times 6` Hamiltonian matrix :math:`H_{ij}=\braketT{\phi_i}{\hat{H}}{\phi_j}`, where :math:`\ket{\phi_i}` is a member of the set of allowed Fock states :math:`\{\ket{2,0,0},\ket{1,1,0},\ket{1,0,1},\ket{0,2,0},\ket{0,1,1},\ket{0,0,2}\}`. Performing these inner products, we find that

.. math::
	H = \begin{bmatrix}
		U & J\sqrt{2} & J\sqrt{2} & 0 & 0 & 0\\
		J\sqrt{2} & 0 & J & J\sqrt{2} & J & 0\\
		J\sqrt{2} & J & 0 & 0 & J & J\sqrt{2}\\
		0 & J\sqrt{2} & 0 & U & J\sqrt{2} & 0\\
		0 & J & J & J\sqrt{2} & 0 & J\sqrt{2}\\
		0 & 0& J\sqrt{2} & 0 & J\sqrt{2} & U
	\end{bmatrix}.

Therefore, using SciPy to perform the matrix exponential :math:`e^{-iHt}` applied to the initial state:

>>> from scipy.linalg import expm
>>> Jr2 = J*np.sqrt(2)
>>> H = np.array(
... 	[[U , Jr2, Jr2, 0  , 0  , 0  ],
... 	[Jr2, 0  , J  , Jr2, J  , 0  ],
... 	[Jr2, J  , 0  , 0  , J  , Jr2],
... 	[0  , Jr2, 0  , U  , Jr2, 0  ],
... 	[0  , J  , J  , Jr2, 0  , Jr2],
... 	[0  , 0  , Jr2, 0  , Jr2, U  ]])
>>> np.abs(expm(-1j*H*1.086)[0])**2
[ 0.0854745, 0.04900244, 0.04900244, 0.30932247, 0.19787567, 0.30932247]

which agrees within reasonable numeric error with the Strawberry Fields simulation results.
