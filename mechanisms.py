import numpy as np

g = 9.81  # kg/ms^2


class SliderCrank:
    def __init__(self, first_link, second_link):
        self.first_link = first_link
        self.second_link = second_link

    def solve(self, theta12):
        r2 = self.first_link
        r3 = self.second_link

        theta13 = np.arcsin(-r2 / r3 * np.sin(theta12))
        s14 = np.sqrt(
            + r2 ** 2
            + r3 ** 2
            + 2 * r2 * r3 * np.cos(theta12) * np.cos(theta13)
            + 2 * r2 * r3 * np.sin(theta12) * np.sin(theta13)
        )

        result = {
            'r2': r2, 'r3': r3, 'theta12': theta12,
            'theta13': theta13, 's14': s14,
        }
        return result


class Scissor:
    def __init__(self, first_link, second_link, first_link_total=None, second_link_total=None, mass=0):
        self.slider_crank = SliderCrank(first_link, second_link)

        if first_link_total is None:
            self.first_link_total = first_link
        else:
            self.first_link_total = first_link_total

        if second_link_total is None:
            self.second_link_total = second_link
        else:
            self.second_link_total = second_link_total

        self.mass = mass

    def solve(self, theta12):
        result = self.slider_crank.solve(theta12)
        s14 = result['s14']

        result['torque'] = -1 * self.mass * g * s14 * (np.cos(theta12) ** -2)
        result['H'] = self.first_link_total * np.sin(theta12)

        result['mass'] = self.mass
        result['p2'] = self.first_link_total
        result['p3'] = self.second_link_total
        return result
