import numpy as np


def angle_between_vectors(vec_0, vec_1):
    dot = np.dot(vec_0, vec_1)
    len_0 = pythagoras(0, 0, vec_0[0], vec_0[1])
    len_1 = pythagoras(0, 0, vec_1[0], vec_1[1])

    return np.arccos(dot / (len_0 * len_1)) * 180 / np.pi


def norm_vector(vec):
    return vec/np.linalg.norm(vec)


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


def density(dist):
    """
    Returns the air density

    :param dist:
    :return: Pressure at height dist
    """
    return 1.28*np.e**-(0.00012*dist)


def a_resis(dist, vel, drag, area):
    """
    Returns the air resistance

    :param dist:
    :param vel:
    :return: Air resistance in Newton
    """
    return ((density(dist)*drag*area)/2)*(vel**2)


def area(mass):
    """
    Returns the average area of an object with mass
    :param mass:
    :return: The average area of the object
    """
    return np.pi * (np.cbrt(((4/3) * np.pi)/(285/mass))**2)  # * np.random.uniform(0.5, 1.5)
