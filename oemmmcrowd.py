import numpy as np
import scipy as sp
import scipy.io as sio
from M3VModel import M3VModel
from sklearn.svm import LinearSVC
class oemmmcrowd(M3VModel):

    def train(self):
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
        TOL = 1e-4
        av = np.ones(Ndom) * self.alpha

        soft_labels = np.zeros( ( self.Ntask, K) )
        mu = np.zeros( (K, self.Ntask) )
        # initializing mu using frequency counts
        if not hasattr(self, 'FreCount'):
            for i in range(self.Ntask):
                neib = self.NeibTask[i]
                labs = L[i, neib]
                for k in range(len(self.LabelDomain)):
                    mu[k, i] = len(np.nonzero(labs == self.LabelDomain[k])[0]) / float(len(labs))
                mu[:, i] = mu[:, i] / mu[:, i].sum()
        else:
            mu = self.FreCount

        soft_labels = mu.transpose()
        for iter in range(self.maxIter):
            # M-step: update workers' confusion matrix
            probd0 = av.reshape(-1,1).repeat(axis=1,repeats= K * self.Nwork)\
                .reshape(Ndom, K, self.Nwork)
            probd = probd0
            for j in range(self.Nwork):
                neib = np.array( self.NeibWork[j] )
                labs = L[neib,j].T
                for ell in range(self.Ndom):
                    dx = neib[labs == self.LabelDomain[ell]]
                    probd[ell,:,j] = probd[ell,:,j] + soft_labels[dx,:].sum(axis=0)
            phi = probd / probd.sum(axis=0).reshape(1,self.Ndom,-1).repeat(axis=0,repeats=self.Ndom)

            #update eta
            xid = np.zeros( ( self.Ntask * (self.Ndom - 1 ), self.Nwork ))
            yid = np.ones( self.Ntask*(self.Ndom -1) )
            count = 0
            for i in range(self.Ntask):
                for j in range(self.Ndom):
                    if j != ( Y[i] - 1 ):
                        xid[count,:] = X[i, Y[i] - 1,:] - X[i, j, :]
                        # if np.random.uniform() > 0.5:
                        #     xid[count,:] = 0 - xid[count,:]
                        #     yid[count] = -1
                        count += 1
            eta0 = np.zeros((1,self.Nwork))
            eta = eta0
            # solve svm subproblem using liblinear
            # if c != 0:
            #     svmmodel = LinearSVC( loss='hinge', C= 2*c )
            #     svmmodel.fit(xid, yid )
            #     wi = svmmodel.coef_
            #     eta = wi * self.v
                # print(eta)

            # solve svm subproblem using gradient descent
            if c != 0:
                for si in range(self.maxIter * 5):
                    deltasi = eta - eta0
                    dx = l * np.ones(self.Ntask * ( self.Ndom - 1 )) - np.dot(eta, xid.transpose())
                    dx = (dx > 0).reshape(-1)
                    deltasi = deltasi - xid[dx,:].sum(axis=0) * ( 2 * c * self.v)
                    deta = deltasi / (si + 1)

                    if ((np.abs(deta)).max() < 1e-2 ):
                        break
                    eta = eta - deta

            #update Y
            try:
                old_soft_labels = soft_labels.copy()
            except:
                old_soft_labels = np.zeros( self.Ntask )

            for i in range( self.Ntask ):
                logprob = np.ones( Ndom ) / float( Ndom )
                for k in range( Ndom ):
                    ds = 0
                    if c != 0:
                        for kk in range( Ndom ):
                            if kk != k:
                                dx = X[i,k,:] - X[i,kk,:]
                                s1 = l - np.dot(eta , dx.reshape(-1,1) )
                                if s1 > ds:
                                    ds = s1
                        logprob[k] = logprob[k] - 2 * c * ds
                    for j in self.NeibTask[i]:
                        logprob[k] = logprob[k] + np.log( phi[L[i,j]-1,k,j] + self.eps )

                prob = np.exp( logprob - logprob.max() )
                prob = prob / prob.sum()
                soft_labels[i,:] = prob.transpose()
                Y[i] = prob.argmax() + 1

            if self.verbose > 0 :
                error_rate = self.cal_error_using_soft_label( soft_labels, self.true_labels )
                print("iter:%s, error_rate:%s, DLL:%s, pi:%s" % (iter, error_rate,1,1))

            # err = np.abs(old_soft_labels - soft_labels).max()
            # if err < self.TOL:


if __name__ == '__main__':
    mm = oemmmcrowd()
    mm.loadData('./datasets/mat/web_data.mat')
    mm.train()
    print("Finished")