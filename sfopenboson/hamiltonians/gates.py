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
r"""Gates module"""
import numpy as np

from openfermion.ops import BosonOperator, QuadOperator


def displacement(alpha, mode=0, hbar=2):
    r"""Returns the Hamiltonian of the displacement operation.

    The time evolution unitary associated with displacement is

    .. math::
        D(\alpha) = \exp( \alpha \ad -\alpha^* \a) = \exp(r (e^{i\phi}\ad -e^{-i\phi}\a))

    where :math:`\alpha=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H = {i}{\hbar}(e^{i\phi}\ad -e^{-i\phi}\a)` and :math:`t=r`.

    Args:
        a (complex): the displacement in the phase space
        mode (int): the qumode on which the operation acts
        hbar (float): the scaling convention chosen in the definition of the quadrature
            operators: :math:`[\x,\p]=i\hbar`
    Returns:
        tuple (BosonOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    if alpha == 0.:
        return BosonOperator(''), 0

    r = np.abs(alpha)
    phi = np.angle(alpha)
    H = BosonOperator('{}^'.format(mode), np.exp(1j*phi))
    H -= BosonOperator('{}'.format(mode), np.exp(-1j*phi))
    return 1j*H*hbar, r


def xdisplacement(x, mode=0):
    r"""Returns the Hamiltonian of the :math:`x` displacement operation.

    The time evolution unitary associated with :math:`x` displacement is

    .. math::
        X(x) = \exp(-ix\p/\hbar)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H = \p` and :math:`t=x`.

    Args:
        x (float): the position displacement in the phase space
        mode (int): the qumode on which the operation acts
    Returns:
        tuple (QuadOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    return QuadOperator('p{}'.format(mode)), x


def zdisplacement(p, mode=0):
    r"""Returns the Hamiltonian of the :math:`p` displacement operation.

    The time evolution unitary associated with :math:`p` displacement is

    .. math::
        X(x) = \exp(ip\x/\hbar)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H =-\x` and :math:`t=p`.

    Args:
        p (float): the position displacement in the phase space
        mode (int): the qumode on which the operation acts
    Returns:
        tuple (QuadOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    return -QuadOperator('q{}'.format(mode)), p


def rotation(phi, mode=0, hbar=2):
    r"""Returns the Hamiltonian of the rotation operation.

    The time evolution unitary associated with rotation is

    .. math::
        R(\phi) = \exp\left(i \phi \ad \a\right)
        =\exp\left(i \frac{\phi}{2} \left(\frac{\x^2+  \p^2}{\hbar}-I\right)\right)

    Therefore, :math:`U=e^{-iHt/\hbar}` where :math:`H = -\hbar\ad\a` and :math:`t=\phi`.

    Args:
        phi (float): the rotation angle
        mode (int): the qumode on which the operation acts
        hbar (float): the scaling convention chosen in the definition of the quadrature
            operators: :math:`[\x,\p]=i\hbar`
    Returns:
        tuple (BosonOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    return -BosonOperator('{}^ {}'.format(mode, mode))*hbar, phi


def squeezing(r, phi=0, mode=0, hbar=2):
    r"""Returns the Hamiltonian of the squeezing operation.

    The time evolution unitary associated with squeezing is

    .. math::
        S(r,\phi) = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H =  \frac{i\hbar}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2}\right)` and :math:`t=r`.

    Args:
        r (float): the squeezing magnitude
        phi (float): the quadrature angle in which the squeezing occurs.
            :math:`\phi=0` corresponds to squeezing in the :math:`\x` quadrature,
            and :math:`\phi=\pi/2` corresponds to squeezing in the
            :math:`\p` quadrature.
        mode (int): the qumode on which the operation acts
        hbar (float): the scaling convention chosen in the definition of the quadrature
            operators: :math:`[\x,\p]=i\hbar`
    Returns:
        tuple (BosonOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    H = BosonOperator('{} {}'.format(mode, mode), np.exp(-1j*phi))
    H -= BosonOperator('{}^ {}^'.format(mode, mode), np.exp(1j*phi))
    return (1j/2)*H*hbar, r


def quadratic_phase(s=1, mode=0):
    r"""Returns the Hamiltonian of the quadratic phase operation.

    The time evolution unitary associated with the quadratic phase is

    .. math::
        P(s) = \exp\left(i  \frac{s}{2 \hbar} \x^2\right)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H = -\x^2/2` and :math:`t=s`.

    Args:
        s (float): the quadratic phase parameter
        mode (int): the qumode on which the operation acts
    Returns:
        tuple (QuadOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    return -QuadOperator('q{} q{}'.format(mode, mode))/2, s


def beamsplitter(theta=np.pi/4, phi=0, mode1=0, mode2=1, hbar=2):
    r"""Returns the Hamiltonian of the beamsplitter operation.

    The time evolution unitary associated with the beamsplitter is

    .. math::
        B(\theta,\phi) = \exp\left(\theta (e^{i \phi} \ad_0 \a_1
             - e^{-i \phi}\a_0 \ad_1) \right)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H(\phi) = {i}{\hbar}\left(e^{i \phi} \ad_0 \a_1 - e^{-i \phi}\a_0 \ad_1\right)`
    and :math:`t=\theta`.

    Args:
        theta (float): transmitivity angle :math:`\theta` where :math:`t=\cos(\theta)`
        phi (float): phase angle :math:`\phi` where :math:`r=e^{i\phi}\sin(\theta)`
        mode1 (int): the first qumode :math:`\a_0` on which the operation acts
        mode2 (int): the second qumode :math:`\a_1` on which the operation acts
        hbar (float): the scaling convention chosen in the definition of the quadrature
            operators: :math:`[\x,\p]=i\hbar`
    Returns:
        tuple (BosonOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    H = BosonOperator('{}^ {}'.format(mode1, mode2), np.exp(1j*(np.pi-phi)))
    H += BosonOperator('{} {}^'.format(mode1, mode2), -np.exp(-1j*(np.pi-phi)))
    return 1j*H*hbar, theta


def two_mode_squeezing(r, phi=0, mode1=0, mode2=1, hbar=2):
    r"""Returns the Hamiltonian of the two-mode squeezing operation.

    The time evolution unitary associated with two-mode squeezing is

    .. math::
        S_2(r,\phi) = \exp\left(r\left(e^{-i\phi}\a_0 \a_1 -e^{i\phi}{\ad_0} \ad_1 \right) \right)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H = {i}{\hbar}\left(e^{-i\phi}\a_0 \a_1 -e^{i\phi}{\ad_0} \ad_1\right)` and :math:`t=r`.

    Args:
        r (float): the squeezing magnitude
        phi (float): the quadrature in which the squeezing occurs.
            :math:`\phi=0` corresponds to squeezing in the :math:`\x` quadrature,
            and :math:`\phi=\pi/2` corresponds to squeezing in the
            :math:`\p` quadrature.
        mode1 (int): the first qumode :math:`\a_0` on which the operation acts
        mode2 (int): the second qumode :math:`\a_1` on which the operation acts
        hbar (float): the scaling convention chosen in the definition of the quadrature
            operators: :math:`[\x,\p]=i\hbar`
    Returns:
        tuple (BosonOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    H = BosonOperator('{} {}'.format(mode1, mode2), np.exp(-1j*(np.pi+phi)))
    H -= BosonOperator('{}^ {}^'.format(mode1, mode2), np.exp(1j*(np.pi+phi)))
    return 1j*H*hbar, r

def controlled_addition(s=1, mode1=0, mode2=1):
    r"""Returns the Hamiltonian of the controlled addition operation.

    The time evolution unitary associated with controlled addition is

    .. math::
        CX(s) = \exp\left( -i \frac{s}{\hbar}\x_0\otimes \p_1 \right)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H =\x_0\otimes \p_1` and :math:`t=s`.

    Args:
        s (float): the controlled addition parameter
        mode1 (int): the first qumode :math:`\a_0` on which the operation acts
        mode2 (int): the second qumode :math:`\a_1` on which the operation acts
    Returns:
        tuple (QuadOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    return QuadOperator('q{} p{}'.format(mode1, mode2)), s


def controlled_phase(s=1, mode1=0, mode2=1):
    r"""Returns the Hamiltonian of the controlled phase operation.

    The time evolution unitary associated with controlled phase is

    .. math::
        CZ(s) = \exp\left( i \frac{s}{\hbar}\x_0\otimes \x_1 \right)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H = -\x_0\otimes \x_1` and :math:`t=s`.

    Args:
        s (float): the controlled addition parameter
        mode1 (int): the first qumode :math:`\a_0` on which the operation acts
        mode2 (int): the second qumode :math:`\a_1` on which the operation acts
    Returns:
        tuple (QuadOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    return -QuadOperator('q{} q{}'.format(mode1, mode2)), s


def cubic_phase(gamma=1, mode=0):
    r"""Returns the Hamiltonian of the cubic phase operation.

    The time evolution unitary associated with the cubic phase is

    .. math::
        V(\gamma) = \exp\left(i  \frac{\gamma}{3 \hbar} \x^3\right)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H = -\x^3/3` and :math:`t=\gamma`.

    Args:
        gamma (float): the cubic phase parameter
        mode (int): the qumode on which the operation acts
    Returns:
        tuple (QuadOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    return -QuadOperator('q{} q{} q{}'.format(mode, mode, mode))/3, gamma


def kerr(kappa=1, mode=0, hbar=2):
    r"""Returns the Hamiltonian of the Kerr operation.

    The time evolution unitary associated with the Kerr gate is

    .. math::
        K(\kappa) = \exp\left(i  \kappa \hat{n}^2 \right)

    Therefore, :math:`U=e^{-iHt/\hbar}` where
    :math:`H = -\hat{n}^2\hbar=-(\ad \a)^2\hbar` and :math:`t=\kappa`.

    Args:
        kappa (float): the Kerr parameter
        mode (int): the qumode on which the operation acts
        hbar (float): the scaling convention chosen in the definition of the quadrature
            operators: :math:`[\x,\p]=i\hbar`
    Returns:
        tuple (BosonOperator, t): tuple containing the Hamiltonian
        representing the operation and the propagation time
    """
    return -BosonOperator('{}^ {} {}^ {}'.format(mode, mode, mode, mode))*hbar, kappa
