import numpy as np
import scipy as sp
import scipy.io as sio
from scipy import special
from crowdsourcingModel import CrowdsourcingModel
from scipy.optimize import minimize


class MiniMax_Entropy_Vector_Crowd_Model(CrowdsourcingModel):
    def train(self):
        # TODO :
        self.priora = 0

        if not hasattr(self, 'prior_tasks'):
            prior_tasks = np.ones((self.Ndom, self.Ntask)) / float(self.Ndom)
        else:
            prior_tasks = self.prior_tasks.copy()

        if not hasattr(self, 'prior_workers'):
            prior_workers = np.ones((self.Ndom, self.Ndom)) + self.priora
        else:
            prior_workers = self.prior_workers.copy()

        mu = np.zeros((self.Ndom, self.Ntask))
        TOL = 1e-3

        # initializing mu using frequency counts
        if not hasattr(self, 'FreCount'):
            for i in range(self.Ntask):
                neib = self.NeibTask[i]
                labs = self.L[i, neib]
                for k in range(len(self.LabelDomain)):
                    mu[k, i] = prior_tasks[k, i] * len(np.nonzero(labs == self.LabelDomain[k])[0]) / float(len(labs))
                mu[:, i] = mu[:, i] / mu[:, i].sum()
        else:
            mu = self.FreCount

        tau = np.zeros((self.Ndom, self.Ndom, self.Ntask))
        sigma = np.zeros((self.Ndom, self.Ndom, self.Nwork))
        logp_task = np.zeros((self.Ndom, self.Ntask))

        # TODO
        inner_maxIter = 1

        for iter in range(self.maxIter):
            # M-step
            for inner_iter in range(inner_maxIter):
                # TODO: if update_alpha
                if 1:
                    for i in range(self.Nwork):
                        neib = self.NeibWork[i]
                        optimal_obj = lambda x: self.getObj(x, neib, self.L[:, i], mu, tau)
                        res = minimize(optimal_obj, sigma[:, :, i], method='L-BFGS-B')
                        sigma[:, :, i] = res.x.reshape(sigma.shape[:2])

                    for j in range(self.Ntask):
                        neib = self.NeibTask[j]
                        optimal_obj = lambda x: self.getObj(x, neib, self.L[j, :], mu, sigma)
                        res = minimize(optimal_obj, tau[:, :, j], method="L-BFGS-B")
                        tau[:, :, j] = res.x.reshape(tau.shape[:2])

            # E - step
            old_mu = mu
            for j in range(self.Ntask):
                neib = self.NeibTask[j]
                mu[:, j] = np.zeros(self.Ndom)
                for i in neib:
                    mu[:, j] = mu[:, j] + self.getP(self.L[j, i] - 1, sigma[:, :, i], tau[:, :, j])
                mu[:, j] = mu[:, j] / mu[:, j].sum()

            error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(mu.transpose(1, 0), self.true_labels)
            print("iter:%s, error_rate:%s, pi:%s" % (-1, error_rate, mu.sum(axis=1) / mu.sum()))

    def getP(self, k, sigma_i, tau_j):
        norm_sum = np.log((np.exp(sigma_i[:, :] + tau_j[:, :])).sum(axis=1))
        return (sigma_i[:, k] + tau_j[:, k]) / norm_sum

    def getObj(self, x, neib, array_k, mu, mat):
        obj = (x ** 2).sum()
        for j in neib:
            obj = obj - (mu[:, j] * self.getP(array_k[j] - 1, x.reshape(mat.shape[:2]), mat[:, :, j])).sum()
        return obj


if __name__ == '__main__':
    m = MiniMax_Entropy_Vector_Crowd_Model()
    m.loadData('./datasets/mat/web_data.mat')
    m.train()
