import numpy as np
import scipy as sp
import scipy.io as sio
from M3VModel import M3VModel
from time import time
import math

class loss_driven_m3v(M3VModel):
    def __init__(self, l=3, c=0.25, n=50, maxIter=50, burnIn=10,
                 v=1, alpha=1, TOL=1e-2, alpha2=35.649, beta2=6,
                 seed=None):
        super(loss_driven_m3v, self).__init__(l=l, c=c, n=n, maxIter=maxIter,
                                              burnIn=burnIn, v=v, alpha=alpha, TOL=TOL,
                                              seed=seed)
        self.alpha2 = alpha2
        self.beta2 = beta2


    def train(self):
        np.random.seed(1)
        print(np.random.rand())
        print(np.random.rand())
        self.initial()
        self.initByMajorityVoting()

        K = self.Ndom
        Ndom = self.Ndom
        L = self.L
        ilm = self.ilm
        eta = self.eta
        phi = self.phi
        X = self.X
        Y = self.Y
        c = self.c
        l = self.l
        S = self.S
        for i in range(self.Ntask):
            S[i] = self.getsi(eta, i, Y[i]-1)

        start_t = time()
        for iter in range(self.maxIter):
            A = self.A0.copy()
            B = self.B0.copy()
            for i in range(self.Ntask):
                dx = X[i,Y[i]-1,:] - X[i, S[i]-1,:]
                dx = dx.reshape(1,-1)
                B = B + np.dot(dx.T,dx) * ilm[i] * c * c
                A = A + ( l * ilm[i] + 1/float(c)) * dx * c *c

            # debug_tmp = np.array(np.dot(A, B.I)).reshape(-1)
            eta = np.random.multivariate_normal( np.array(np.dot(A, B.I)).reshape(-1), B.I ).reshape(1,-1) \
                  + self.eps
            # print(eta)
            #-- copy is necessary because probd0 cannot be change --
            probd = self.probd0.copy()
            for i in range(self.Ntask):
                for j in self.NeibTask[i]:
                    probd[L[i,j]-1, Y[i]-1, j] = probd[ L[i,j]-1, Y[i]-1, j ] + 1

            for i in range(K):
                for j in range(self.Nwork):
                    phi[:,i,j] = np.random.dirichlet(probd[:,i,j], 1) + self.eps
                    # print( "phi", phi [:,i,j])
                    # print( "probd", probd[:,i,j])

            for i in range(self.Ntask):
                dx = X[i, Y[i]-1, :] - X[i, S[i]-1, :]
                aczetai = abs( c * ( l - np.dot(eta,dx.T)) + self.eps)
                ilm[i] = np.random.wald( np.linalg.inv(aczetai), 1)

            randomIdx = np.array(range(self.Ntask))
            np.random.shuffle(randomIdx)
            # print(randomIdx[:15])
            for i in randomIdx:
                logprob = np.zeros( ( K, 1) )
                for k in range(K):
                    # if ilm[i] != 0
                    if abs( ilm[i] ) > self.eps:
                        dx = X[i,k,:] - X[i, self.getsi(eta, i, k) - 1, :]
                        logprob[k] = -0.5 * ilm[i] * ( 1.0/ilm[i] + c * ( l - np.dot(eta,dx.T) ) )**2
                    for j in self.NeibTask[i]:
                        # TODO:it is right?? ??
                        logprob[k] = logprob[k] + np.log( phi[ L[i,j]-1, k, j ] + self.eps )

                prob = np.exp( logprob - logprob.max() )
                prob = prob/prob.sum()
                prob_sample = np.random.multinomial(1,prob.reshape(-1))
                prob_nnz = np.nonzero(prob_sample > 0)
                Y[i] = int(prob_nnz[0]) + 1
                S[i] = self.getsi(eta, i, Y[i]-1)

            if iter > self.burnIn:
                for i in range(self.Ntask):
                    self.ans_soft_labels[i,Y[i]-1] = self.ans_soft_labels[i,Y[i]-1] + 1
                self.phic = self.phic + phi
                self.etak[self.etak_count,:] = eta[0,:]
                self.etak_count += 1


            if self.verbose > 0 and iter > self.burnIn:
                            if iter == 47:
                                a = 1
                            ans_soft_labelst = self.ans_soft_labels/( self.etak_count )
                            error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(ans_soft_labelst,self.true_labels)
                            end_t = time()
                            print("iter:%s, error_rate:%s, totaltime:%s" % (iter, error_rate, end_t - start_t))
                            # print( self.ans_soft_labels )

        ans_soft_labelst = self.ans_soft_labels / (self.etak_count)
        error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(ans_soft_labelst,self.true_labels)
        end_t = time()
        print("Final: iter:%s, error_rate:%s, totaltime:%s" % (iter, error_rate, end_t - start_t))

