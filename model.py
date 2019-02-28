import numpy as np
from scipy.spatial.distance import pdist, squareform
import physics

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation


class OrbitModel:

    def __init__(self,
                 init_state=[8.371 * (10 ** 6), 0, 0, 6904],
                 #bounds=[-2, 2, -2, 2],
                 m_1=40*(10**3),
                 m_2=5.972*(10**24),
                 r_2=6378000,
                 G=6.67384*(10**-11)):
        #self.bounds = bounds
        self.init_state = np.asarray(init_state, dtype=float)
        self.m_1 = m_1
        self.m_2 = m_2
        self.r_2 = r_2
        self.state = self.init_state.copy()
        self.time_elapsed = 0
        self.G = G

    def step(self, dt, push=False):
        """step once by dt seconds"""
        self.time_elapsed += dt

        """update positions"""
        self.state[:2] += dt * self.state[2:]

        """friction"""
        #self.state[2:]

        """calculate gravity"""
        r = physics.pythagoras(self.state[0], self.state[1], 0, 0)
        gravity_F = physics.gravity(self.G, self.m_1, self.m_2, r)

        a = gravity_F/self.m_1

        norm = (0 - self.state[:2])/np.linalg.norm(0 - self.state[:2])

        gravity_vector = norm * a * dt

        """add gravity"""
        self.state[2:] += gravity_vector

        push_vector = [0, 0]
        if push:
            push_force = 1

            norm = (0 - self.state[2:]) / np.linalg.norm(0 - self.state[2:])

            push_vector = norm * push_force/self.m_1 * dt
            """add gravity"""
            self.state[2:] += push_vector

        return self.state[:2], [gravity_vector, push_vector]




