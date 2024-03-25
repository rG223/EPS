import torch
import numpy as np


def get_problem(name, n_dim=10, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        'dtlz2': DTLZ2,
        'convex_dtlz2':Convex_DTLZ2,
        're21': RE21,
        're23': RE23,
        're31': RE31,
        're33': RE33,
        're34': RE34,
        're36': RE36,
        're37': RE37,
        'dtlz4': DTLZ4,
        'dtlz5': DTLZ5,
        'dtlz7': DTLZ7,
        're32': RE32,
        'zdt3': ZDT3,
    }
    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](n_dim=n_dim)


class F1():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            yi = x[:, i - 1] - torch.pow(2 * x[:, 0] - 1, 2)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F2():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            theta = 1.0 + 3.0 * (i - 2) / (n - 2)
            yi = x[:, i - 1] - torch.pow(x[:, 0], 0.5 * theta)
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F3():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0.0
        count1 = count2 = 0.0

        for i in range(2, n + 1):
            xi = x[:, i - 1]
            yi = xi - (torch.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n) + 1) / 2
            yi = yi * yi

            if i % 2 == 0:
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0]))

        objs = torch.stack([f1, f2]).T

        return objs


class F4():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - 0.8 * x[:, 0] * torch.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:, 0] * torch.cos(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F5():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - 0.8 * x[:, 0] * torch.sin(4.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - 0.8 * x[:, 0] * torch.cos((4.0 * np.pi * x[:, 0] + i * np.pi / n) / 3)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class F6():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        sum1 = sum2 = 0
        count1 = count2 = 0

        for i in range(2, n + 1):
            xi = -1.0 + 2.0 * x[:, i - 1]

            if i % 2 == 0:
                yi = xi - (0.3 * x[:, 0] ** 2 * torch.cos(12.0 * np.pi * x[:, 0] + 4 * i * np.pi / n) + 0.6 * x[:,
                                                                                                              0]) * torch.sin(
                    6.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                yi = yi * yi
                sum2 = sum2 + yi
                count2 = count2 + 1.0
            else:
                yi = xi - (0.3 * x[:, 0] ** 2 * torch.cos(12.0 * np.pi * x[:, 0] + 4 * i * np.pi / n) + 0.6 * x[:,
                                                                                                              0]) * torch.cos(
                    6.0 * np.pi * x[:, 0] + i * np.pi / n)
                yi = yi * yi
                sum1 = sum1 + yi
                count1 = count1 + 1.0

        f1 = (1 + 1.0 / count1 * sum1) * x[:, 0]
        f2 = (1 + 1.0 / count2 * sum2) * (1.0 - torch.sqrt(x[:, 0] / (1 + 1.0 / count2 * sum2)))

        objs = torch.stack([f1, f2]).T

        return objs


class VLMOP1():
    def __init__(self, n_dim=1):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([-2.0]).float()
        self.ubound = torch.tensor([4.0]).float()
        self.nadir_point = [4, 4]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1 = torch.pow(x[:, 0], 2)
        f2 = torch.pow(x[:, 0] - 2, 2)

        objs = torch.stack([f1, f2]).T

        return objs


class VLMOP2():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0]).float()
        self.ubound = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        n = x.shape[1]

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1 = 1 - torch.exp(-torch.sum((x - 1 / np.sqrt(n)) ** 2, axis=1))
        f2 = 1 - torch.exp(-torch.sum((x + 1 / np.sqrt(n)) ** 2, axis=1))

        objs = torch.stack([f1, f2]).T

        return objs


class VLMOP3():
    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([-3.0, -3.0]).float()
        self.ubound = torch.tensor([3.0, 3.0]).float()
        self.nadir_point = [10, 60, 1]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1, x2 = x[:, 0], x[:, 1]

        f1 = 0.5 * (x1 ** 2 + x2 ** 2) + torch.sin(x1 ** 2 + x2 ** 2)
        f2 = (3 * x1 - 2 * x2 + 4) ** 2 / 8 + (x1 - x2 + 1) ** 2 / 27 + 15
        f3 = 1 / (x1 ** 2 + x2 ** 2 + 1) - 1.1 * torch.exp(-x1 ** 2 - x2 ** 2)

        objs = torch.stack([f1, f2, f3]).T

        return objs


class DTLZ2():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1, 1]

    def evaluate(self, x):
        n = x.shape[1]
        x = x + torch.normal(mean=0, std=0.06, size=(1, self.n_dim)).cuda()
        x[x < 0] = 0
        x[x > 1] = 1
        sum1 = torch.sum(torch.stack([torch.pow(x[:, i] - 0.5, 2) for i in range(2, n)]), axis=0)
        g = sum1

        f1 = (1 + g) * torch.cos(x[:, 0] * np.pi / 2) * torch.cos(x[:, 1] * np.pi / 2)
        f2 = (1 + g) * torch.cos(x[:, 0] * np.pi / 2) * torch.sin(x[:, 1] * np.pi / 2)
        f3 = (1 + g) * torch.sin(x[:, 0] * np.pi / 2)

        objs = torch.stack([f1, f2, f3]).T

        return objs

class Convex_DTLZ2():
    def __init__(self, n_dim=6):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.zeros(n_dim).float()
        self.ubound = torch.ones(n_dim).float()
        self.nadir_point = [1, 1, 1]

    def evaluate(self, x):
        n = x.shape[1]
        x = x + torch.normal(mean=0, std=0.06, size=(1, self.n_dim)).cuda()
        x[x < 0] = 0
        x[x > 1] = 1
        sum1 = torch.sum(torch.stack([torch.pow(x[:, i] - 0.5, 2) for i in range(2, n)]), axis=0)
        g = sum1

        f1 = (1 + g) * torch.cos(x[:, 0] * np.pi / 2) * torch.cos(x[:, 1] * np.pi / 2)
        f2 = (1 + g) * torch.cos(x[:, 0] * np.pi / 2) * torch.sin(x[:, 1] * np.pi / 2)
        f3 = (1 + g) * torch.sin(x[:, 0] * np.pi / 2)

        objs = torch.stack([f1**4, f2**4, f3**2]).T

        return objs

class RE21():
    def __init__(self, n_dim=4):
        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma

        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val]).float()
        self.ubound = torch.ones(n_dim).float() * 3 * tmp_val
        self.nadir_point = [2886.3695604236013, 0.039999999999998245]

    def evaluate(self, x):
        F = 10.0
        E = 2.0 * 1e5
        L = 200.0

        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        f1 = L * ((2 * x[:, 0]) + np.sqrt(2.0) * x[:, 1] + torch.sqrt(x[:, 2]) + x[:, 3])
        f2 = ((F * L) / E) * (
                    (2.0 / x[:, 0]) + (2.0 * np.sqrt(2.0) / x[:, 1]) - (2.0 * np.sqrt(2.0) / x[:, 2]) + (2.0 / x[:, 3]))

        f1 = f1
        f2 = f2

        objs = torch.stack([f1, f2]).T

        return objs


class RE23():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 2
        self.lbound = torch.tensor([1, 1, 10, 10]).float()
        self.ubound = torch.tensor([100, 100, 200, 240]).float()
        self.nadir_point = [5852.05896876, 1288669.78054]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = 0.0625 * torch.round(x[:, 0])
        x2 = 0.0625 * torch.round(x[:, 1])
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = (0.6224 * x1 * x3 * x4) + (1.7781 * x2 * x3 * x3) + (3.1661 * x1 * x1 * x4) + (19.84 * x1 * x1 * x3)
        f1 = f1.float()

        # Original constraint functions
        g1 = x1 - (0.0193 * x3)
        g2 = x2 - (0.00954 * x3)
        g3 = (np.pi * x3 * x3 * x4) + ((4.0 / 3.0) * (np.pi * x3 * x3 * x3)) - 1296000

        g = torch.stack([g1, g2, g3])
        z = torch.zeros(g.shape).cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f2 = torch.sum(g, axis=0).to(torch.float64)

        objs = torch.stack([f1, f2]).T

        return objs



class RE31():
    def __init__(self, n_dim=3):
        self.problem_name = 'RE31'
        self.n_obj = 3
        self.n_dim = 3
        self.n_constraints = 0
        self.n_original_constraints = 3
        self.nadir_point = [500, 9000000, 20000000]
        self.lbound = torch.tensor([0.00001, 0.00001, 1.0]).float()
        self.ubound = torch.tensor([100, 100, 3.0]).float()

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        # First original objective function
        f1 = x1 * torch.sqrt(16.0 + (x3 * x3)) + x2 * torch.sqrt(1.0 + x3 * x3)
        # Second original objective function
        f2 = (20.0 * torch.sqrt(16.0 + (x3 * x3))) / (x1 * x3)

        # Constraint functions
        g1 = 0.1 - f1
        g2 = 100000.0 - f2
        g3 = 100000 - ((80.0 * torch.sqrt(1.0 + x3 * x3)) / (x3 * x2))
        g = torch.stack([g1, g2, g3])
        z = torch.zeros(g.shape).cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)
        f3 = torch.sum(g, axis=0).to(torch.float64)
        objs = torch.stack([f1, f2, f3]).T
        return objs


class RE33():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([55, 75, 1000, 11]).float()
        self.ubound = torch.tensor([80, 110, 3000, 20]).float()
        self.nadir_point = [5.3067, 3.12833430979, 25.0]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f2 = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        # Reformulated objective functions
        g1 = (x2 - x1) - 20.0
        g2 = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g3 = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / torch.pow((x2 * x2 - x1 * x1), 2)
        g4 = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0

        g = torch.stack([g1, g2, g3, g4])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)

        f3 = torch.sum(g, axis=0).float()

        objs = torch.stack([f1, f2, f3]).T

        return objs


class RE34():
    def __init__(self, n_dim=5):
        self.problem_name = 'RE34'
        self.n_obj = 3
        self.n_dim = 5
        self.lbound = torch.tensor([1, 1, 1, 1, 1]).float()
        self.ubound = torch.tensor([3, 3, 3, 3, 3]).float()

        self.nadir_point = [1695, 10, 0.3]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()
        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]
        x5 = x[:, 4]

        f1 = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (4.4559504 * x5)
        f2 = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (
                0.0861 * x1 * x5) + (0.3628 * x2 * x4) - (0.1106 * x1 * x1) - (0.3437 * x3 * x3) + (
                     0.1764 * x4 * x4)
        f3 = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (
                0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)

        objs = torch.stack([f1, f2, f3]).T

        return objs


class RE36():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([12, 12, 12, 12]).float()
        self.ubound = torch.tensor([60, 60, 60, 60]).float()
        self.nadir_point = [5.931, 56.0, 0.355720675227]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        # First original objective function
        f1 = torch.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        # Second original objective function (the maximum value among the four variables)
        l = torch.stack([x1, x2, x3, x4])
        f2 = torch.max(l, dim=0)[0]

        g1 = 0.5 - (f1 / 6.931)

        g = torch.stack([g1])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)
        f3 = g[0]

        objs = torch.stack([f1, f2, f3]).T

        return objs


class RE37():
    def __init__(self, n_dim=4):
        self.n_dim = n_dim
        self.n_obj = 3
        self.lbound = torch.tensor([0, 0, 0, 0]).float()
        self.ubound = torch.tensor([1, 1, 1, 1]).float()
        self.nadir_point = [0.98949120096, 0.956587924661, 0.987530948586]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound

        xAlpha = x[:, 0]
        xHA = x[:, 1]
        xOA = x[:, 2]
        xOPTT = x[:, 3]

        # f1 (TF_max)
        f1 = 0.692 + (0.477 * xAlpha) - (0.687 * xHA) - (0.080 * xOA) - (0.0650 * xOPTT) - (0.167 * xAlpha * xAlpha) - (
                    0.0129 * xHA * xAlpha) + (0.0796 * xHA * xHA) - (0.0634 * xOA * xAlpha) - (0.0257 * xOA * xHA) + (
                         0.0877 * xOA * xOA) - (0.0521 * xOPTT * xAlpha) + (0.00156 * xOPTT * xHA) + (
                         0.00198 * xOPTT * xOA) + (0.0184 * xOPTT * xOPTT)
        # f2 (X_cc)
        f2 = 0.153 - (0.322 * xAlpha) + (0.396 * xHA) + (0.424 * xOA) + (0.0226 * xOPTT) + (0.175 * xAlpha * xAlpha) + (
                    0.0185 * xHA * xAlpha) - (0.0701 * xHA * xHA) - (0.251 * xOA * xAlpha) + (0.179 * xOA * xHA) + (
                         0.0150 * xOA * xOA) + (0.0134 * xOPTT * xAlpha) + (0.0296 * xOPTT * xHA) + (
                         0.0752 * xOPTT * xOA) + (0.0192 * xOPTT * xOPTT)
        # f3 (TT_max)
        f3 = 0.370 - (0.205 * xAlpha) + (0.0307 * xHA) + (0.108 * xOA) + (1.019 * xOPTT) - (0.135 * xAlpha * xAlpha) + (
                    0.0141 * xHA * xAlpha) + (0.0998 * xHA * xHA) + (0.208 * xOA * xAlpha) - (0.0301 * xOA * xHA) - (
                         0.226 * xOA * xOA) + (0.353 * xOPTT * xAlpha) - (0.0497 * xOPTT * xOA) - (
                         0.423 * xOPTT * xOPTT) + (0.202 * xHA * xAlpha * xAlpha) - (0.281 * xOA * xAlpha * xAlpha) - (
                         0.342 * xHA * xHA * xAlpha) - (0.245 * xHA * xHA * xOA) + (0.281 * xOA * xOA * xHA) - (
                         0.184 * xOPTT * xOPTT * xAlpha) - (0.281 * xHA * xAlpha * xOA)

        objs = torch.stack([f1, f2, f3]).T

        return objs


class UF6:
    def __init__(self, n_var=10, n_obj=3):
        self.n_dim = 30
        self.n_obj = 2
        self.Alpha = 0.05
        self.K = 5
        self.lbound = -torch.ones(1, self.n_dim).float()
        self.lbound[0] = self.lbound[0] + 1
        self.ubound = torch.ones(1, self.n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, X):  # 目标函数
        D = self.n_dim
        J1 = torch.arange(2, D, 2).to(X.device)
        J2 = torch.arange(1, D, 2).to(X.device)

        Y = X - torch.sin(
            6 * torch.pi * torch.tile(X[:, 0].reshape(-1, 1), (1, D)) + torch.tile(
                torch.arange(1, D + 1) * torch.pi / D,
                (X.shape[0], 1)).to(X.device))

        f1 = X[:, 0] + torch.maximum(torch.zeros(1, X.shape[0]).to(X.device),
                                     2 * (1 / 4 + 0.1) * torch.sin(4 * np.pi * X[:, 0])) + (
                     2 / len(J1)) * (
                     4 * torch.sum(Y[:, J1] ** 2, axis=1) - 2 * torch.prod(
                 torch.cos(20 * Y[:, J1] * torch.pi / torch.sqrt(J1)),
                 axis=1) + 2)

        f2 = 1 - X[:, 0] + torch.maximum(torch.zeros(1, X.shape[0]).to(X.device),
                                         2 * (1 / 4 + 0.1) * torch.sin(4 * torch.pi * X[:, 0])) + (
                     2 / len(J2)) * (
                     4 * torch.sum(Y[:, J2] ** 2, axis=1) - 2 * torch.prod(
                 torch.cos(20 * Y[:, J2] * torch.pi / torch.sqrt(J2)),
                 axis=1) + 2)

        objs = torch.stack([f1[0], f2[0]]).T

        return objs


def obj_func(X_, g, n_obj, alpha=1):
    f = []

    for i in range(0, n_obj):
        _f = (1 + g)
        _f *= torch.prod(torch.cos(torch.pow(X_[:, :X_.shape[1] - i], alpha) * torch.pi / 2.0), axis=1)
        if i > 0:
            _f *= torch.sin(torch.pow(X_[:, X_.shape[1] - i], alpha) * torch.pi / 2.0)

        f.append(_f)

    f = torch.column_stack(f)
    return f


def g2(X_M):
    return torch.sum(torch.square(X_M - 0.5), axis=1)


class DTLZ4:
    def __init__(self, n_dim):
        self.n_obj = 3
        self.n_dim = n_dim
        self.k = self.n_dim - self.n_obj + 1
        self.alpha = 1
        self.d = 100
        self.lbound = torch.zeros(1, self.n_dim).float()
        self.ubound = torch.ones(1, self.n_dim).float()
        self.nadir_point = [1, 1, 1]

    def evaluate(self, x):
        out = {}
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = g2(X_M)
        out["F"] = obj_func(X_, g, self.n_obj, alpha=self.alpha)

        return out["F"]


class DTLZ5:
    def __init__(self, n_dim):
        self.n_obj = 3
        self.n_dim = n_dim
        self.n_var = n_dim
        self.k = self.n_dim - self.n_obj + 1
        self.lbound = torch.zeros(1, self.n_dim).float()
        self.ubound = torch.ones(1, self.n_dim).float()
        self.nadir_point = [1, 1, 1]

    def evaluate(self, x):
        out = {}
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = g2(X_M)

        theta = 1 / (2 * (1 + g[:, None])) * (1 + 2 * g[:, None] * X_)
        theta = torch.column_stack([x[:, 0], theta[:, 1:]])
        out["F"] = obj_func(theta, g, self.n_obj)
        return out['F']


class DTLZ7:
    def __init__(self, n_dim):
        self.n_obj = 3
        self.n_dim = n_dim
        self.k = self.n_dim - self.n_obj + 1
        self.lbound = torch.zeros(1, self.n_dim).float()
        self.ubound = torch.ones(1, self.n_dim).float()
        self.nadir_point = [1, 1, 6]

    def evaluate(self, x):
        out = {}
        f = []
        for i in range(0, self.n_obj - 1):
            f.append(x[:, i])
        f = torch.column_stack(f)

        g = 1 + 9 / self.k * torch.sum(x[:, -self.k:], axis=1)
        h = self.n_obj - torch.sum(f / (1 + g[:, None]) * (1 + torch.sin(3 * torch.pi * f)), axis=1)

        out["F"] = torch.column_stack([f, (1 + g) * h])

        return out["F"]


class RE32():
    def __init__(self, n_dim=4):
        self.problem_name = 'RE32'
        self.n_dim = 4
        self.n_obj = 3
        self.lbound = torch.tensor([0.125, 0.1, 0.1, 0.125]).float()
        self.ubound = torch.tensor([5, 10, 10, 5]).float()
        self.nadir_point = [40, 20, 5e+8]

    def evaluate(self, x):
        if x.device.type == 'cuda':
            self.lbound = self.lbound.cuda()
            self.ubound = self.ubound.cuda()

        x = x * (self.ubound - self.lbound) + self.lbound
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x4 = x[:, 3]

        P = 6000
        L = 14
        E = 30 * 1e6

        # // deltaMax = 0.25
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000

        # First original objective function
        f1 = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        # Second original objective function
        f2 = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)

        # Constraint functions
        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + torch.pow((x1 + x3) / 2.0, 2)
        R = torch.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + torch.pow((x1 + x3) / 2.0, 2)
        J = 2 * torch.sqrt(torch.tensor(2)) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J
        tauDash = P / (torch.sqrt(torch.tensor(2)) * x1 * x2)
        tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
        tau = torch.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = 4.013 * E * torch.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * torch.sqrt(torch.tensor(E / (4 * G)))
        PC = tmpVar * (1 - tmpVar2)

        g1 = tauMax - tau
        g2 = sigmaMax - sigma
        g3 = x4 - x1
        g4 = PC - P
        g = torch.stack([g1, g2, g3, g4])
        z = torch.zeros(g.shape).float().cuda().to(torch.float64)
        g = torch.where(g < 0, -g, z)
        f3 = torch.sum(g, axis=0).float()

        objs = torch.stack([f1, f2, f3]).T

        return objs


class ZDT3:
    def __init__(self, n_dim):
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = torch.zeros(1, self.n_dim).float()
        self.ubound = torch.ones(1, self.n_dim).float()
        self.nadir_point = [1, 1]

    def evaluate(self, x):
        # f1 = torch.cos(x[:, 0])**2 + 0.2
        # f2 = 1.3+torch.sin(x[:, 1])**2-torch.cos(x[:, 0])-0.1*torch.sin(22*torch.pi*torch.cos(x[:, 0])**2)**5 # -0.5
        f1 = x[:, 0]
        c = torch.sum(x[:, 1:], axis=1)
        g = 1.0 + 9.0 * c / (self.n_dim - 1)
        f2 = g * (1 - torch.pow(f1 * 1.0 / g, 0.5) - (f1 * 1.0 / g) * torch.sin(10 * torch.pi * f1))
        objs = torch.stack([f1, f2]).T
        return objs

class ZDT4:
    def __init__(self, n_dim):
        super().__init__(n_var)
        self.n_obj = 2
        self.n_dim = n_dim
        self.lbound = -5 * np.ones(self.n_var)
        self.lbound[0] = 0.0
        self.ubound = 5 * np.ones(self.n_var)
        self.ubound[0] = 1.0
        self.nadir_point = [1, 1]
    def evaluate(self, x):
        f1 = x[:, 0]
        g = 1.0
        g += 10 * (self.n_dim - 1)
        for i in range(1, self.n_dim):
            g += x[:, i] * x[:, i] - 10.0 * torch.cos(4.0 * torch.pi * x[:, i])
        h = 1.0 - torch.sqrt(f1 / g)
        f2 = g * h
        objs = torch.stack([f1, f2]).T
        return objs


class ZDT5:

    def __init__(self, m=11, n=5):
        self.m = m
        self.n = n
        self.n_obj = 2
        self.n_dim = 30 + n * (m - 1)
        self.nadir_point = [1, 1]
    def evaluate(self, x):
        x = x.astype(float)

        _x = [x[:, :30]]
        for i in range(self.m - 1):
            _x.append(x[:, 30 + i * self.n: 30 + (i + 1) * self.n])

        u = torch.column_stack([x_i.sum(axis=1) for x_i in _x])
        v = (2 + u) * (u < self.n) + 1 * (u == self.n)
        g = v[:, 1:].sum(axis=1)

        f1 = 1 + u[:, 0]
        f2 = g * (1 / f1)


        objs = torch.stack([f1, f2]).T
        return objs
