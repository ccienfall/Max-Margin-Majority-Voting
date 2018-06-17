import numpy as np
import scipy as sp
import scipy.io as sio
from scipy import special
from M3VModel import M3VModel
from sklearn.svm import LinearSVC

class emmmcrowd(M3VModel):

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
        for iter in self.maxIter:
            # M-step: update workers' confusion matrix
            probd0 = av.reshape(-1,1).repeat(axis=1,repeats= K * self.Nwork)\
                .reshape(Ndom, K, self.Nwork)
            probd = probd0
            for j in range(self.Nwork):
                neib = np.array( self.NeibWork[j] )
                labs = L[neib,j].T
                for ell in range(self.Ndom):
                    dx = neib[labs == self.LabelDomain[ell]]
                    probd[ell,:,j] = probd[ell,:,j] + self.ans_soft_labels[dx,:].sum(axis=0)
            phi = probd / probd.sum(axis=0).reshape(1,self.Ndom,-1).repeat(axis=0,repeats=self.Ndom)

            #update eta
            xid = np.zeros( ( self.Ntask * (self.Ndom - 1 ), self.Nwork ))
            yid = np.ones( self.Ntask*(self.Ndom -1) )
            count = 0
            for i in range(self.Ntask):
                for j in range(self.Ndom):
                    if j != ( Y[i] - 1 ):
                        xid[count,:] = X[i, Y[i],:] - X[i, j, :]
            if c != 0:
                svmmodel = LinearSVC( loss='hinge', C= 2*c )
                wi = svmmodel.class_weight
                eta = wi * self.v

            #update Y
            try:
                old_soft_labels = soft_labels
            except:
                old_soft_labels = np.zeros( self.Ntask )

            randomIdx = range(self.Ntask)
            np.random.shuffle(randomIdx)
            for i in randomIdx:
                logprob = np.ones( Ndom ) / float( Ndom )
                for k in range( Ndom ):
                    ds = 0
                    if c != 0:
                        for kk in range( Ndom ):
                            if kk != k:
                                dx = np.zeros( self.Nwork )
                                dx[:] = X[i,k,:] - X[i,kk,:]
                                s1 = 1 - eta * dx
                                if s1 > ds:
                                    ds = s1
                        logprob[k] = logprob[k] - 2 * c * ds
                    for j in self.NeibTask[i]:
                        logprob[k] = logprob[k] + np.log( phi[L[i,j]-1,k,j] + self.eps )

                prob = np.exp( logprob - logprob.max() )
                prob = prob / prob.sum()
                soft_labels[i,:] = prob.transpose()
                Y[i] = prob.argmax() + 1

            # ap = np.zeros( Ndom)
            # ii = 0
            # for j in range( self.Nwork ):
            #     for k in range( self.Ndom ):
            #         ii += 1
            #         for d in range( Ndom ):
            #             ap[d] += np.log( phi[d,k,j] + self.eps)
            #
            # for kkk in range(10):
            #     aaa = special.psi( sum(av) )
            #     for d in range( Ndom ):
            #         av[d] = invpsi[ap[d] + aaa ]
            #     alpha = sum(av) / float(Ndom)
            # NLL = 0
            # NLL1 = 0
            # NLL2 = 0
            # for j in range( self.Nwork ):
            #     for i in range( self.NeibWork[j] ):
            #         NLL1 -= np.log( phi[L[i,j]-1, :, j].sum() + self.eps )
            #
            # for k in range( Ndom ):
            #     for j in range( self.Nwork ):
            #         for d in range( Ndom ):
            #             NLL -= probd[d,k,j] * np.log( phi[d,k,j] + self.eps)
            #             NLL2 = NLL2 - ( probd[d,k,j] - alpha ) * np.log(phi[d,k,j] + self.eps)



            if self.verbose > 1:
                error_rate = self.cal_error_using_soft_label(soft_labels, self.true_labels )
                print("iter:%s, error_rate:%s" % ( iter, error_rate ) )


if __name__ == '__main__':

    print("Finished")