# -*- coding: utf-8 -*-
# author: jzh

# 使用方法：
# 共有三个参数，第一个参数输入维数（待优化参数个数），第二个参数输入一个二维列表，代表每个维数的取值范围，第三个参数输入评估函数
# main_loop返回这些参数的取值

import random
import math


class individual:

    def __init__(self, n, pos, fit):
        self.n = n  # dimension
        self.pos = pos  # position
        self.fit = fit

    def __repr__(self):
        return str(self.pos)


class rand:

    def random(self):
        return random.random()

    def levy_flight(self):
        return random.gauss(0, 1) / random.random()**(1 / 1.5)

    def normal(self):
        return random.gauss(0, 1)

    def get_m(self, n, m):
        x = set()
        while len(x) < m:
            x.add(random.randint(0, n - 1))
        return x


class nutcracker:

    def __init__(self, n, lim, eval):
        self.n = n  # dimension
        self.lim = lim  # limitation
        self.NP = 10  # Population size
        self.Tmax = 100
        self.Pa1 = 0.1
        self.Pa2 = 0.1
        self.Prp = 0.5  # haven't get in the paper
        self.delta = 0.1
        self.rand = rand()
        self.eval = eval
        self.RP1, self.RP2 = [[None] * self.n for _ in range(self.NP)], [[None] * self.n for _ in range(self.NP)]
        self.best_lim = None

    def pop_init(self):
        rand = self.rand
        self.pop = []
        self.new_pop = []
        for i in range(self.NP):
            temp = []
            for j in range(self.n):
                temp.append((self.lim[j][1] - self.lim[j][0]) * rand.random() + self.lim[j][0])
            self.pop.append(individual(self.n, temp, self.eval(temp)))
            self.new_pop.append(individual(self.n, [None] * self.n, None))
        self.best_idx = 0
        self.best_fit = self.pop[0].fit
        for i in range(1, self.NP):
            if self.pop[i].fit < self.best_fit:
                self.best_fit = self.pop[i].fit
                self.best_idx = i

    def main_loop(self):
        self.pop_init()
        rand = self.rand
        self.env = False
        for t in range(self.Tmax):
            self.l = 1 - (t+1) / self.Tmax
            self.t = t + 1
            if rand.random() < rand.random():  # Foraging and storage strategy
                self.phase_1()
                self.update()
            else:  # cache-search and recovery strategy
                self.gen_RP()
                self.phase_2()
                self.update()
            print(t, self.best_fit)
            if self.best_lim and abs(self.best_fit - self.best_lim) < 0.0000001:
                break
        return self.pop[self.best_idx].pos

    def phase_1(self):
        rand = self.rand
        mean = [0] * self.n
        for j in range(self.n):
            for i in range(self.NP):
                mean[j] += self.pop[i].pos[j]
            mean[j] /= self.NP
        if rand.random() < self.Pa1:  # Forage
            for i in range(self.NP):
                a, b, c = rand.get_m(self.NP, 3)
                gamma = rand.levy_flight()
                Tau1, Tau2 = rand.random(), rand.random()
                r1, r2, r3 = rand.random(), rand.random(), rand.random()
                if r1 > r2 and r1 > r3:
                    Mu = rand.random()
                elif r2 > r3:
                    Mu = rand.normal()
                else:
                    Mu = rand.levy_flight()
                if Tau1 > Tau2:
                    if self.t <= self.Tmax << 1:
                        for j in range(self.n):
                            self.new_pop[i].pos[j] = mean[j] + gamma * (self.pop[a].pos[j] - self.pop[b].pos[j]) + Mu * (
                                rand.random()**2 * self.lim[j][1] - self.lim[j][0]
                            )
                    else:
                        for j in range(self.n):
                            self.new_pop[i].pos[j] = self.pop[c].pos[j] + Mu * (self.pop[a].pos[j] - self.pop[b].pos[j]) + Mu * (
                                rand.random()**2 * self.lim[j][1] - self.lim[j][0]
                            ) * (r1 < self.delta)
                else:
                    for j in range(self.n):
                        self.new_pop[i].pos[j] = self.pop[i].pos[j]
        else:  # Storage
            for i in range(self.NP):
                Tau1, Tau2, Tau3 = rand.random(), rand.random(), rand.random()
                a, b = rand.get_m(self.NP, 2)
                r1, r2, r3 = rand.random(), rand.random(), rand.random()
                lam = abs(rand.levy_flight())
                if r1 > r2 and r1 > r3:
                    Mu = rand.random()
                elif r2 > r3:
                    Mu = rand.normal()
                else:
                    Mu = rand.levy_flight()
                if Tau1 < Tau2 and Tau1 < Tau3:
                    for j in range(self.n):
                        self.new_pop[i].pos[j] = self.pop[i].pos[j] + Mu * lam * (
                            self.pop[self.best_idx].pos[j] - self.pop[i].pos[j]
                        ) + r1 * (self.pop[a].pos[j] - self.pop[b].pos[j])
                elif Tau2 < Tau3:
                    for j in range(self.n):
                        self.new_pop[i].pos[j] = self.pop[self.best_idx].pos[j] + Mu * (self.pop[a].pos[j] - self.pop[b].pos[j])
                else:
                    for j in range(self.n):
                        self.new_pop[i].pos[j] = self.pop[self.best_idx].pos[j] * self.l

    def gen_RP(self):
        rand = self.rand
        RP = []
        for j in range(self.n):
            RP.append((self.lim[j][1] - self.lim[j][0]) * rand.random() + self.lim[j][0])
        for i in range(self.NP):
            r1, r2 = rand.random(), rand.random()
            if r1 < r2:
                Alpha = (self.t / self.Tmax)**(2 / self.t)
            else:
                Alpha = (1 - self.t / self.Tmax)**(2 * self.t / self.Tmax)
            xita = rand.random() * math.pi
            a, b = rand.get_m(self.NP, 2)
            Tau3 = rand.random()
            for j in range(self.n):
                self.RP1[i][j] = self.pop[i].pos[j] + Alpha * math.cos(xita) * (self.pop[a].pos[j] - self.pop[b].pos[j])
            for j in range(self.n):
                self.RP2[i][j] = self.pop[i].pos[j]
                if random.random() < self.Prp:
                    self.RP2[i][j] += Alpha * math.cos(xita) * ((self.lim[j][1] - self.lim[j][0]) * Tau3 + self.lim[j][0])

    def phase_2(self):
        rand = self.rand
        mean = [0] * self.n
        for j in range(self.n):
            for i in range(self.NP):
                mean[j] += self.pop[i].pos[j]
            mean[j] /= self.NP
        for i in range(self.NP):
            if rand.random() < self.Pa2:  # EQ.17
                fit1 = self.eval(self.RP1[i])
                fit2 = self.eval(self.RP2[i])
                if fit1 < fit2 and fit1 < self.pop[i].fit:
                    self.new_pop[i].pos = self.RP1[i].copy()
                elif fit2 < self.pop[i].fit:
                    self.new_pop[i].pos = self.RP2[i].copy()
                else:
                    self.new_pop[i].pos = self.pop[i].pos.copy()
            else:  # EQ.16
                Tau7, Tau8 = rand.random(), rand.random()
                if Tau7 < Tau8:
                    Tau3, Tau4 = rand.random(), rand.random()
                    r1, r2 = rand.random(), rand.random()
                    if Tau3 < Tau4:
                        self.new_pop[i].pos = self.pop[i].pos.copy()
                    else:
                        for j in range(self.n):
                            self.new_pop[i].pos[j] = self.pop[i].pos[
                                j] + r1 * (self.pop[self.best_idx].pos[j] - self.pop[i].pos[j]) + r2 * (self.RP1[i][j] - mean[j])
                else:
                    Tau5, Tau6 = rand.random(), rand.random()
                    r1, r2 = rand.random(), rand.random()
                    if Tau5 < Tau6:
                        self.new_pop[i].pos = self.pop[i].pos.copy()
                    else:
                        for j in range(self.n):
                            self.new_pop[i].pos[j] = self.pop[i].pos[
                                j] + r1 * (self.pop[self.best_idx].pos[j] - self.pop[i].pos[j]) + r2 * (self.RP2[i][j] - mean[j])

    def update(self):
        for i in range(self.NP):
            for j in range(self.n):
                self.new_pop[i].pos[j] = min(self.lim[j][1], max(self.lim[j][0], self.new_pop[i].pos[j]))
            self.new_pop[i].fit = self.eval(self.new_pop[i].pos)
            if self.new_pop[i].fit < self.pop[i].fit:
                self.pop[i], self.new_pop[i] = self.new_pop[i], self.pop[i]
                if self.pop[i].fit < self.best_fit:
                    self.best_fit = self.pop[i].fit
                    self.best_idx = i


if __name__ == '__main__':

    def rastrigin(x):  # test function, best point is [0]*n, best fiitness is 0
        ans = 10 * len(x)
        for i in range(len(x)):
            ans += x[i]**2 - 10 * math.cos(2 * math.pi * x[i])
        return ans

    m = 30
    lim = []
    for i in range(m):
        lim.append((-5.12, 5.12))
    instance = nutcracker(m, lim, rastrigin)
    instance.main_loop()
    print(instance.best_fit)
