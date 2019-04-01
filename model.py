import numpy as np
from scipy.spatial.distance import pdist, squareform
import physics

import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation


class OrbitModel:

    def __init__(self,
                 init_state=[8.371 * (10 ** 6), 0, 0, 6904],
                 m_1=40,
                 drag=0.8,
                 pulse_strength=0.06,
                 custom_height=False):

        self.m_1 = m_1
        self.m_2 = 5.972*(10**24)
        self.r_2 = 6378000
        self.time_elapsed = 0
        self.G = 6.67384*(10**-11)
        self.drag = drag # np.random.uniform(0.04, 1.15)
        self.area = physics.area(m_1)
        self.pulse_strength = pulse_strength
        self.score = 0

        if custom_height:
            self.init_state = np.float64([self.r_2 + custom_height, 0, 0, np.sqrt(self.G*self.m_2/(self.r_2+custom_height))])
        else:
            self.init_state = np.float64(np.asarray(init_state, dtype=float))
        self.state = self.init_state.copy()

    def step(self, dt, push=False, train=False):
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

        norm = physics.norm_vector(0 - self.state[:2])

        gravity_vector = norm * a * dt

        """add gravity"""
        self.state[2:] += gravity_vector

        """ calculate air resistance """
        dist = r - self.r_2
        resis_F = physics.a_resis(dist, np.linalg.norm(self.state[2:]), self.drag, self.area)

        resis_a = resis_F/self.m_1
        resis_norm = (0 - self.state[2:])/np.linalg.norm(0 - self.state[2:])
        resis_vector = resis_norm * resis_a * dt
        self.state[2:] += resis_vector

        push_vector = [0, 0]
        if push:
            push_force = self.pulse_strength

            if push % 2 == 0:
                norm = (0 - self.state[2:]) / np.linalg.norm(0 - self.state[2:])
            else:
                norm = (self.state[2:]) / np.linalg.norm(self.state[2:])

            push_vector = norm * push_force/self.m_1 * dt
            """add gravity"""
            self.state[2:] += push_vector

        x1 = self.state[0]
        y1 = self.state[1]

        x2 = np.cos(2 * np.pi / 86400 * self.time_elapsed)
        y2 = np.sin(2 * np.pi / 86400 * self.time_elapsed)

        h = physics.pythagoras(self.state[0], self.state[1], 0, 0) - self.r_2

        a = np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)

        v = np.linalg.norm(self.state[2:])

        if h < 0:
            x_2 = np.cos(2 * np.pi / 86400 * self.time_elapsed)
            y_2 = np.sin(2 * np.pi / 86400 * self.time_elapsed)
            score = 180 - physics.angle_between_vectors(self.state[:2], [x_2, y_2])
            if train:
                return np.array([a, h, v]), score, True
            else:
                return self.state[:2]

        if train:
            test_model = OrbitModel(init_state=self.state, m_1=self.m_1, drag=self.drag, pulse_strength=self.pulse_strength)

            if self.time_elapsed % 600 == 1:

                i = 0
                while True:
                    pos = test_model.step(dt)
                    test_h = physics.pythagoras(pos[0], pos[1], 0, 0) - self.r_2
                    if test_h < 0:
                        x_2 = np.cos(2 * np.pi / 86400 * test_model.time_elapsed)
                        y_2 = np.sin(2 * np.pi / 86400 * test_model.time_elapsed)
                        self.score = 180 - physics.angle_between_vectors(pos, [x_2, y_2])
                        break
                    # elif i == 43200 - 1:
                    #     self.score = 0
                    #     break
                    # print(i, end="\r")
                    i += 1
                # print(i)
                    # print(str(test_h) + "                 " + str(self.score), end="\r")

            return np.array([a, h, v]), self.score, False
        else:
            return self.state[:2], push_vector

    def reset(self):
        self.time_elapsed = 0
        self.state = self.init_state.copy()

        x1 = self.state[0]
        y1 = self.state[1]

        x2 = np.cos(2 * np.pi / 86400 * self.time_elapsed)
        y2 = np.sin(2 * np.pi / 86400 * self.time_elapsed)

        h = physics.pythagoras(self.state[0], self.state[1], 0, 0) - self.r_2
        a = np.arctan2(x1 * y2 - y1 * x2, x1 * x2 + y1 * y2)
        v = np.linalg.norm(self.state[2:])

        return np.array([a, h, v])






