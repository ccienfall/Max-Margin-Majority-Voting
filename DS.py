import numpy as np
import scipy as sp
import scipy.io as sio
from scipy import special
from crowdsourcingModel import CrowdsourcingModel

import warnings
warnings.filterwarnings('error')

class DS(CrowdsourcingModel):

    def __init__(self, l=3, c=0.25, n=50, maxIter=50, burnIn=10, v=1, alpha=1, hard_predict = False):
        super().__init__( l, c, n, maxIter, burnIn, v, alpha)
        self.hard_predict = hard_predict

    def train(self):
        #TODO:
        # self.priora = 0.1

        L = self.L

        #TODO:
        if hasattr(self,'partial_truth'):
            partial_truth = self.partial_truth
        else:
            partial_truth = [[],[]]

        partial_dx = partial_truth[0]
        partial_array = np.ones( (self.Ndom, len(partial_dx) ) ) / float(self.Ndom)
        for i in range(len(partial_dx)):
            partial_array[:,i] = self.eps
            partial_array[partial_truth[1],i] = 1 - self.eps
            partial_array[:,i] = partial_array[:,i] / partial_array[:,i].sum()

        other_dx = np.array(range(self.Ntask))

        #set default prior parameters
        if not hasattr(self,'prior_tasks'):
            prior_tasks = np.ones( (self.Ndom, self.Ntask)) / float(self.Ndom)
        else:
            prior_tasks = self.prior_tasks.copy()

        if not hasattr(self,'prior_workers'):
            prior_workers = np.ones( ( self.Ndom, self.Ndom)) + self.priora
        else:
            prior_workers = self.prior_workers.copy()
        phi= np.ones( ( self.Ndom, self.Ndom, self.Nwork ) )
        mu = np.zeros( (self.Ndom, self.Ntask) )
        # add by Changjian, for pi update. 2017/8/3
        pi = np.zeros( self.Ndom )

        #initializing mu using frequency counts
        if not hasattr(self,'FreCount'):
            for i in range(self.Ntask):
                neib = self.NeibTask[i]
                labs = L[i,neib]
                for k in range(len(self.LabelDomain)):
                    mu[k,i] = prior_tasks[k,i] * len(np.nonzero(labs==self.LabelDomain[k])[0])/float(len(labs))
                mu[:,i] = mu[:,i] / mu[:,i].sum()
            mu[:,partial_dx] = partial_array
        else:
            mu = self.FreCount

        error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(mu.transpose(1, 0), self.true_labels)
        print("iter:%s, error_rate:%s, soft_error_rate:%s, pi:%s" % (-1, error_rate, soft_error_rate, mu.sum(axis=1)/mu.sum() ))

        #EM algorithms
        err = float('inf')
        for iter in range(self.maxIter):
            # M-step: updating workers' confusion matrix(phi)
            for j in range(self.Nwork):
                if( iter == (self.maxIter-1) and (j == self.Nwork - 1)):
                    debug = 1
                neib = np.array(self.NeibWork[j])
                labs = L[neib,j].T
                # TODO: Is this the right way to add prior
                # fixed, it is right.
                phi[:,:,j] = (prior_workers - 1) + self.eps
                for ell in range(self.Ndom):
                    dx = neib[labs==self.LabelDomain[ell]]
                    # phi dimensions means : 1-d:true labels, 2-d labels labeled by works, 3-d works' id
                    phi[:,ell,j] = phi[:,ell,j] + mu[:,dx].sum(axis=1)
            phi = phi / phi.sum(axis=1).reshape(self.Ndom,1,-1).repeat(axis=1,repeats=self.Ndom)
            pi = mu.sum( axis=1 )
            # pi = np.ones(mu.shape[0])
            # pi = np.array([ (self.true_labels==(i+1)).sum() for i in range(self.Ndom)])
            pi = pi / pi.sum()

            #E-step: Updating tasks' posterior probabilities(mu)
            old_mu = mu
            for i in other_dx:
                neib = np.array(self.NeibTask[i])
                labs = L[i,neib]
                tmp = 0
                for ell in range(self.Ndom):
                    jdx = neib[labs==self.LabelDomain[ell]]
                    tmp = tmp + np.log(phi[:,ell,jdx]).sum(axis=1)

                # add for pi updating
                tmp = tmp + np.log(pi)

                mu[:,i] = prior_tasks[:,i] * np.exp(tmp - tmp.max())
                mu[:, i] = mu[:, i] / mu[:, i].sum()

            if self.hard_predict:
                mu = self.predict(mu)

            if self.verbose > 0:
                error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(mu.transpose(1,0),self.true_labels)
                NLL = self.cal_NLL( phi, mu,pi, self.L,prior_workers )
                print("iter:%s, error_rate:%s, soft_error:%s, NLL:%s, pi:%s" % (iter, error_rate, soft_error_rate,NLL,pi))
        # print(mu.transpose())
        # print( self.predict(mu).transpose() )
        return error_rate

    def predict(self, mu):
        mu = mu.argmax( axis=0 ) + 1
        mu = mu.reshape(1,-1).repeat( axis=0, repeats=self.Ndom )
        tmp = np.array( range(1, self.Ndom+1)).reshape(-1,1).repeat( axis=1, repeats=mu.shape[1] )
        Ones = np.zeros( (mu.shape) )
        Ones[mu == tmp] = 1
        return Ones

    def cal_NLL(self, phi, mu, pi, L, prior_workers):
        NLL = 0
        # mu = self.predict(mu)
        alpha = np.zeros((self.Ndom, self.Ndom, self.Nwork))
        for j in range(self.Nwork):
            neib = np.array(self.NeibWork[j])
            labs = L[neib, j].T
            alpha[:, :, j] = (prior_workers - 1) + self.eps
            for ell in range(self.Ndom):
                dx = neib[labs == self.LabelDomain[ell]]
                alpha[:, ell, j] = alpha[:, ell, j] + mu[:, dx].sum(axis=1)
        NLL = - ( np.log(phi) * alpha).sum()
        # NLL = NLL - ( mu.sum(axis=1) + np.log(pi) ).sum()
        return NLL

# create by Changjian, 2017/8/4
class FullBayesianDS( DS ):
    def train(self):

        L = self.L
        #TODO:
        if hasattr(self,'partial_truth'):
            partial_truth = self.partial_truth
        else:
            partial_truth = [[],[]]

        partial_dx = partial_truth[0]
        partial_array = np.ones( (self.Ndom, len(partial_dx) ) ) / float(self.Ndom)
        for i in range(len(partial_dx)):
            partial_array[:,i] = self.eps
            partial_array[partial_truth[1],i] = 1 - self.eps
            partial_array[:,i] = partial_array[:,i] / partial_array[:,i].sum()

        other_dx = np.array(range(self.Ntask))

        #set default prior parameters
        if not hasattr(self,'prior_tasks'):
            prior_tasks = np.ones( (self.Ndom, self.Ntask)) / float(self.Ndom)
        else:
            prior_tasks = self.prior_tasks.copy()
        if not hasattr(self,'prior_workers'):
            prior_workers = np.ones( ( self.Ndom, self.Ndom)) + self.priora
        else:
            prior_workers = self.prior_workers.copy()
        alpha = np.ones( ( self.Ndom, self.Ndom, self.Nwork ) )
        mu = np.zeros( (self.Ndom, self.Ntask) )
        # add for pi update
        pi = np.zeros( self.Ndom )

        prior_pi = np.ones( self.Ndom )

        #initializing mu using frequency counts
        if not hasattr(self,'FreCount'):
            for i in range(self.Ntask):
                neib = self.NeibTask[i]
                labs = L[i,neib]
                for k in range(len(self.LabelDomain)):
                    mu[k,i] = prior_tasks[k,i] * len(np.nonzero(labs==self.LabelDomain[k])[0])/float(len(labs))
                mu[:,i] = mu[:,i] / mu[:,i].sum()
            mu[:,partial_dx] = partial_array
        else:
            mu = self.FreCount

        error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(mu.transpose(1, 0), self.true_labels)
        print("iter:%s, error_rate:%s, pi:%s" % (-1, error_rate, mu.sum(axis=1)/mu.sum() ))

        #EM algorithms
        err = float('inf')
        for iter in range(self.maxIter):
            # M-step: updating workers' confusion matrix's distribution, namely it's parameters : alpha
            for j in range(self.Nwork):
                neib = np.array(self.NeibWork[j])
                labs = L[neib,j].T
                # TODO: Is this the right way to add prior
                # fixed, it is right.
                alpha[:,:,j] = (prior_workers - 1) + self.eps
                for ell in range(self.Ndom):
                    dx = neib[labs==self.LabelDomain[ell]]
                    # alpha dimensions means : 1-d:true labels, 2-d labels labeled by works, 3-d works' id
                    alpha[:,ell,j] = alpha[:,ell,j] + mu[:,dx].sum(axis=1)

            #TODO: assumming that pi is uniform distribution and fixed now
            pi = mu.sum( axis=1 ) + prior_pi
            # # pi = np.ones(mu.shape[0])
            # # pi = np.array([ (self.true_labels==(i+1)).sum() for i in range(self.Ndom)])
            # pi = pi / pi.sum()

            #E-step: Updating tasks' posterior probabilities(mu)
            old_mu = mu
            E_phi = self.ExpectionOfLogPhi(alpha)
            E_pi = self.ExpectionOfLogPi(pi)
            for i in other_dx:
                neib = np.array(self.NeibTask[i])
                labs = L[i,neib]
                tmp = 0
                for ell in range(self.Ndom):
                    jdx = neib[labs==self.LabelDomain[ell]]
                    tmp = tmp + E_phi[:,ell,jdx].sum( axis = 1 )
                # TODO: Pi is support now
                # tmp = tmp + E_pi
                mu[:,i] = prior_tasks[:,i] * np.exp(tmp - tmp.max())
                mu[:, i] = mu[:, i] / mu[:, i].sum()

            if self.verbose > 0:
                error_rate, soft_error_rate, error_L1, erroe_L2 = self.cal_error_using_soft_label(mu.transpose(1,0),self.true_labels)
                NLL = self.cal_NLL( alpha, mu,pi, self.L,prior_workers )
                print("iter:%s, error_rate:%s, soft_error_rate:%s, NLL:%s, pi:%s" % (iter, error_rate,soft_error_rate, NLL,pi))
        return error_rate

    def ExpectionOfLogPhi(self, alpha):
        E_phi = np.zeros( alpha.shape )
        for n in range( self.Nwork ):
            for d in range( self.Ndom ):
                psi_of_sum = special.psi( alpha[d,:,n].sum() )
                for k in range( self.Ndom ):
                    E_phi[d,k,n] = special.psi( alpha[d,k ,n] ) - psi_of_sum
        return E_phi

    def ExpectionOfLogPi(self, pi):
        E_pi = np.zeros( pi.shape )
        psi_of_sum = special.psi( pi.sum() )
        for n in range( self.Ndom ):
            E_pi[n] = special.psi( pi[n]) - psi_of_sum
        return E_pi

    # def CoefOfDirichlet(self, alpha):
    #     GammaOfSum = special.gamma( alpha.sum() )
    #     GammaOfProduct = 1
    #     for i in alpha:
    #         GammaOfProduct *= special.gamma(i)
    #     return GammaOfSum / GammaOfProduct

    def gammaToList(self, a):
        L = []
        while( a > 3 ):
            L.append(a - 1)
            a = a - 1
        L.append( special.gamma(a) )
        return L

    def LogCoefOfDirichlet(self, alpha):
        GammaOfSum = self.gammaToList( alpha.sum() )
        GammaOfProduct = []
        for i in alpha:
            GammaOfProduct = GammaOfProduct + self.gammaToList( i )
        return np.log( np.array( GammaOfSum )).sum() - np.log( np.array( GammaOfProduct)).sum()

    def cal_NLL(self, phi, mu, pi, L, prior_workers):
        NLL = 0
        # mu = self.predict(mu)
        alpha = np.zeros((self.Ndom, self.Ndom, self.Nwork))
        for j in range(self.Nwork):
            neib = np.array(self.NeibWork[j])
            labs = L[neib, j].T
            alpha[:, :, j] = (prior_workers - 1) + self.eps
            for ell in range(self.Ndom):
                dx = neib[labs == self.LabelDomain[ell]]
                alpha[:, ell, j] = alpha[:, ell, j] + mu[:, dx].sum(axis=1)
        # NLL = - ( np.log(phi) * alpha).sum()
        for n in range( alpha.shape[2] ):
            for d in range( self.Ndom ):
                try:
                    NLL -= - ( self.LogCoefOfDirichlet(alpha[d,:,n] ))
                except:
                    print( alpha[d,:,n])
                    exit(0)
        # TODO ; it is needed if using pi
        # NLL = NLL - ( mu.sum(axis=1) + np.log(pi) ).sum()
        return NLL

if __name__ == '__main__':
    # d = DS()
    # d.loadData('../CrowdSVM_code/datasets/age_data.mat')
    # # print d.LabelDomain
    # d.train()
    import os
    import sys
    from logger import Logger
    # data_name = ['flower','bluebird','age','web','temporal','rte','syn','duck','dog']
    data_name = ['age']
    min = 1000
    min_d = None

    log = Logger('log_file.log')
    sys.stdout = log

    for name in data_name:
        priora = 1

        print( "FullBatesianDS Model!" )
        d =FullBayesianDS()
        data_path = os.path.join('./datasets/mat/',name+'_data.mat')
        d.loadData(data_path)
        d.priora = priora
        # tmp_min = d.train()

        print( "DS Model!" )
        d = DS()
        data_path = os.path.join('./datasets/mat/', name + '_data.mat')
        d.loadData(data_path)
        d.priora = priora
        tmp_min = d.train()

        # test = np.array([   1.49796078, 15.43609135 , 142.63483547 ,  50.53258951 , 4.02829067])
        # print( d.LogCoefOfIdrichlet(test) )

    log.close()