import numpy as np


def pythagoras(x_1, y_1, x_2, y_2):
    return np.sqrt((x_1-x_2)**2 + (y_1 - y_2)**2)


def gravity(G, m_1, m_2, r):
    """
    Returns the gravitational force

    :param G: Gravitational Acceleration in m/s^2
    :param m_1: Mass of m_1 in kg
    :param m_2: Mass of m_2 in kg
    :param r: Distance between m_1 and m_2 in kilometer
    :return: Force in Newton
    """
    return G * m_1 * m_2 / (r**2)

