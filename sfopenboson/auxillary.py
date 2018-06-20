# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Auxillary functions
===================

This module contains auxiliary functions for extracting the parameters of
quadratic and Bose-Hubbard Hamiltonians.

Quadratic Hamiltonian functions
-------------------------------

.. autosummary::
    quadratic_coefficients

Bose Hubbard functions
----------------------

.. autosummary::
    BoseHubbardError
    extract_tunneling
    extract_onsite_chemical
    extract_dipole
    trotter_layer

Code details
------------
"""
from itertools import groupby

import numpy as np

from openfermion.utils import is_hermitian


def quadratic_coefficients(operator):
    r"""Return the quadratic coefficient matrix representing a Gaussian Hamiltonian.

    A Gaussian Hamiltonian is any combination of quadratic operators
    that can be written in quadratic form:

    .. math:: H = \frac{1}{2}\mathbf{r}^T A\mathbf{r} + \mathbf{r}^T \mathbf{d}

    where :math:`A\in\mathbb{R}^{2N\times 2N}` is a symmetric matrix,
    :math:`\mathbf{d}\in\mathbb{R}^{2N}` is a real vector, and
    :math:`\mathbf{r} = (\x_1,\dots,\x_N,\p_1,\dots,\p_N)` is the vector
    of quadrature operators in :math:`xp`-ordering.

    This function accepts a bosonic Gaussian Hamiltonian, and returns the
    matrix :math:`A` and vector :math:`\mathbf{d}` representing the
    quadratic and linear coefficients.

    Args:
        operator (QuadOperator): a bosonic Gaussian Hamiltonian
    Returns:
        tuple (A, d): a tuple contains a 2Nx2N real symmetric numpy array,
        and a length-2N real numpy array, where N is the number of modes
        the operator acts on.
    """
    if not operator.is_gaussian():
        raise ValueError("Hamiltonian must be Gaussian "
                         "(quadratic in the quadrature operators).")

    if not is_hermitian(operator):
        raise ValueError("Hamiltonian must be Hermitian.")

    num_modes = max([op[0] for term in operator.terms for op in term])+1
    A = np.zeros([2*num_modes, 2*num_modes])
    d = np.zeros([2*num_modes])
    for term, coeff in operator.terms.items():
        c = coeff.real
        if len(term) == 2:
            if term[0][1] == term[1][1]:
                if term[0][1] == 'q':
                    A[term[0][0], term[1][0]] = c
                elif term[0][1] == 'p':
                    A[num_modes+term[0][0], num_modes+term[1][0]] = c
            else:
                if term[0][1] == 'q':
                    A[term[0][0], num_modes+term[1][0]] = c
                elif term[0][1] == 'p':
                    A[num_modes+term[0][0], term[1][0]] = c
        elif len(term) == 1:
            if term[0][1] == 'q':
                d[num_modes+term[0][0]] = -c
            elif term[0][1] == 'p':
                d[term[0][0]] = c

    A += A.T
    return A, d


class BoseHubbardError(ValueError):
    """Custom error function for invalid Bose-Hubbard Hamiltonians."""

    def with_traceback(self, tb):
        # pylint: disable=useless-super-delegation
        """This method sets argument ``tb`` as the new traceback for the exception
        and returns the exception object. See the
        `Python documentation <https://docs.python.org/3/library/exceptions.html#BaseException.with_traceback>`_
        for more details.
        """
        # this method is overwritten simply due to a bug in Sphinx,
        # which automatically pulls this method into the documentation
        # even if it is not present. By overwriting, we at least get
        # to modify the docstring presented.
        return super().with_traceback(tb)


def extract_tunneling(H):
    """Extracts the tunneling terms from Bose-Hubbard Hamiltonians.

    Args:
        H (BosonOperator): A Bose-Hubbard Hamiltonian.

    Returns:
        list: Returns a length-2 list of the form ``[[(i, j),...], t]``
        where ``[(i, j),...]`` is a list containing pairs of modes
        that are entangled by beamsplitters due to tunneling,
        and ``t`` (float) is the tunneling coefficient.
    """
    BS = {}

    # created a sorted list containing terms with two operators,
    # indexed by the mode of each operator
    tmp = sorted([((o[0][0], o[1][0]), o, c) for o, c in H.terms.items() if len(o) == 2])

    # group into terms
    for key, group in groupby(tmp, lambda x: x[0]):

        # consider only multi-mode terms
        if key[0] != key[1]:
            # copy iterable to a list
            group_list = list(group)

            # check only one
            if len(group_list) != 2:
                raise BoseHubbardError

            # check ladders are of the right form
            # of bi^bj and bj^bi
            ladders = np.array(list(np.array(group_list)[:, 1]))[:, :, 1]
            ladders = set([tuple(i) for i in ladders.tolist()])
            if ladders != {(0, 1), (1, 0)}:
                raise BoseHubbardError

            # check coefficients are the same
            if group_list[0][-1] != group_list[1][-1]:
                raise BoseHubbardError

            BS[key] = group_list[0][-1]
            t = BS[key]

    # check all beamsplitters have the same tunneling coefficient
    if len(set(BS.values())) != 1:
        raise BoseHubbardError

    return [list(BS.keys()), -t]


def extract_onsite_chemical(H):
    """Extracts the onsite interactions and chemical potential terms
    from Bose-Hubbard Hamiltonians.

    Args:
        H (BosonOperator): A Bose-Hubbard Hamiltonian.

    Returns:
        tuple(list, list): Returns a tuple containing two lists; the
        first list is the onsite interaction, and the second list is
        the chemical potential.
        Each list is a length-2 list of the form ``[[i,j,...], t]``
        where ``[i,j,...]`` is a list containing modes operated on,
        and t is the onsite coefficient U or chemical potential mu.
    """
    # Note: we have to consider both onsite and chemical potential terms
    # at the same time, since they share the same structure.
    onsite = {}
    chemical = {}
    mu = 0
    U = 0

    # created a sorted list containing terms where the first two operators
    # act on the *same* mode. It is indexed by this mode.
    tmp = sorted([(o[0][0], o, c) for o, c in H.terms.items() if o[0][0] == o[1][0]])

    # remove multi mode terms
    tmp = sorted([i for i in tmp if len(set(np.array(i[1]).T[0])) == 1])

    # iterate through elements grouped by the indexed mode
    for key, group in groupby(tmp, lambda x: x[0]):

        # copy iterable to a list
        group_list = list(group)

        # check correct number of terms
        # should be one bk^ bk and one bk^ bk bk^ bk if onsite is present,
        # or just one bk^ bk for chemical potential only.
        if len(group_list) > 2:
            raise BoseHubbardError

        if len(group_list) == 1 and len(group_list[0][1]) != 2:
            raise BoseHubbardError

        coeff_dict = {}

        for term in group_list:
            # check correct ordering of ladder operators
            tmp2 = np.array(term[1])[:, 1]
            expected = np.zeros(tmp2.shape)
            expected[::2] = 1
            if not np.all(tmp2 == expected):
                raise BoseHubbardError

            # if the ordering is correct, save the coefficient,
            # with the key as the length of the term (2 or 4).
            coeff_dict[tmp2.shape[0]] = term[-1]

        # check coefficients are of the right form
        term_lengths = sorted(list(coeff_dict.keys()))
        if term_lengths == [2, 4]:
            # on-site interactions present
            U = coeff_dict[4]*2
            onsite[key] = U

            # chemical potential present
            mu = -coeff_dict[2]-onsite[key]/2
            chemical[key] = mu

        elif term_lengths == [2]:
            # only chemical potential present
            mu = -coeff_dict[2]
            chemical[key] = mu
        else:
            raise BoseHubbardError

    # check all have the same U and mu
    if len(set(onsite.values())) > 1:
        raise BoseHubbardError
    if len(set(chemical.values())) > 1:
        raise BoseHubbardError

    onsite = [list(onsite.keys()), U] if onsite else []

    if mu == 0:
        chemical = []
    else:
        chemical = [list(chemical.keys()), mu] if chemical else []

    return onsite, chemical


def extract_dipole(H):
    """Extracts the dipole terms from Bose-Hubbard Hamiltonians.

    Args:
        H (BosonOperator): A Bose-Hubbard Hamiltonian.

    Returns:
        list: Returns a length-2 list of the form ``[[(i, j),...], V]``
        where ``[(i, j),...]`` is a list containing pairs of modes
        that are entangled due to nearest-neighbour interactions,
        and ``V`` (float) is the dipole coefficient.
    """
    dipole = {}

    # extract all length 4 multimode terms
    tmp = [term for term in H.terms.items() if len(term[0]) == 4]

    # created a sorted list containing terms where the first operator
    # and the third operator act on the *same* mode. It is indexed by these modes.
    tmp = sorted([((o[0][0], o[2][0]), o, c) for o, c in tmp if o[0][0] != o[2][0]])

    if not tmp:
        return []

    # iterate through elements grouped by the indexed modes
    for key, group in groupby(tmp, lambda x: x[0]):
        # copy iterable to a list
        group_list = list(group)

        # check only one term is present
        if len(group_list) != 1:
            raise BoseHubbardError

        # check ladders are of the right form bi^bi bj^ bj
        ladders = np.array(list(np.array(group_list)[:, 1]))[:, :, 1]
        ladders = set([tuple(i) for i in ladders.tolist()])
        if ladders != {(1, 0, 1, 0)}:
            raise BoseHubbardError

        # extract the dipole coefficient
        V = group_list[0][-1]
        dipole[key] = V

    # check all terms have the same V
    if len(set(dipole.values())) != 1:
        raise BoseHubbardError

    return [list(dipole.keys()), V]


def trotter_layer(H, t, k):
    """Returns a single Trotter layer for a Bose-Hubbard Hamiltonian.

    Args:
        H (BosonOperator): A Bose-Hubbard Hamiltonian.
        t (float): the time propagation duration.
        k (int): the number of products in the truncated
            Lie product formula.

    Returns:
        dict: A dictionary containing the items:
            * ``'BS': (theta, phi, modes)`` corresponding to beamsplitters
              with parameters ``(theta, phi)`` acting on the list of modes provided.
            * ``'K': (kappa, modes)`` corresponding to Kerr gates with parameter
              ``kappa`` acting on the list of modes provided.
            * ``'R': (r, modes)`` corresponding to rotation gates with parameter
              ``r`` acting on the list of modes provided.
    """
    try:
        BS = extract_tunneling(H)
        onsite, chemical = extract_onsite_chemical(H)
        dipole = extract_dipole(H)
    except BoseHubbardError:
        raise BoseHubbardError("Hamiltonian is not a valid Bose-Hubbard model. "
                               "Are you generating it using the bose_hubbard function?")

    layer_dict = {}

    theta = -t*BS[1]/k
    phi = np.pi/2
    layer_dict['BS'] = (theta, phi, BS[0])

    U = 0
    if onsite:
        U = onsite[1]
        kappa = -t*U/(2*k)
        layer_dict['K'] = (kappa, onsite[0])

    mu = 0
    if chemical:
        mu = chemical[1]
        r = t*(0.5*U+mu)/(2*k)
        layer_dict['R'] = (r, chemical[0])
    elif onsite:
        layer_dict['R'] = (t*U/(2*k), onsite[0])

    if dipole:
        raise BoseHubbardError("Nearest-neighbour or dipole interactions "
                               "not currently supported.")

    return layer_dict
