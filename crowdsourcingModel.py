import numpy as np
import scipy as sp
import scipy.io as sio
import math

class CrowdsourcingModel(object):
    def __init__(self, l=3, c=0.25, n=50, maxIter=50, burnIn=10, v=1, alpha=1, TOL=1e-2, seed = None):
        self.l = l
        self.c = c
        self.n = n
        self.maxIter = maxIter
        self.burnIn = burnIn
        self.v = v
        self.alpha = alpha
        self.eps = 2e-6
        self.verbose = 1

    def loadData(self,filename):
        mat = sio.loadmat(filename)
        self.L = mat['L']
        try:
            self.L = self.L.toarray()
        except:
            self.L = self.L
        self.L = self.L.astype(np.int)
        self.true_labels = mat['true_labels'].reshape(-1).astype(np.int)
        self.Ntask ,self.Nwork = self.L.shape
        self.Ndom = len( set([ i for i in self.true_labels.reshape(-1)]) )
        self.LabelDomain = np.unique(self.L[self.L!=0])
        self.Ndom = len(self.LabelDomain)
        # print( self.LabelDomain )
        # exit(0)
        self.NeibTask = []
        for i in range(self.Ntask):
            tmp = [ nt for nt in range(self.Nwork) if self.L[i,nt] > 0 ]
            self.NeibTask.append(tmp)
        self.NeibWork = []
        for j in range(self.Nwork):
            tmp = [nw for nw in range(self.Ntask) if self.L[nw,j] > 0 ]
            self.NeibWork.append(tmp)
        self.LabelTask = []
        for i in range(self.Ntask):
            tmp = [ self.L[i,nt] for nt in self.NeibTask[i]]
            self.LabelTask.append(tmp)
        self.LabelWork = []
        for j in range(self.Nwork):
            tmp = [ self.L[nw,j] for nw in self.NeibWork[j]]
            self.LabelWork.append(tmp)

    def cal_error_using_soft_label(self,mu,true_labels):
        # print(mu[:5,:])
        # print(true_labels[:5])
        index = ( true_labels > 0 )
        mu = mu[index,:]
        true_labels = true_labels[index]
        # print(mu.shape, true_labels.shape)
        # print(mu[:5, :])
        # print(true_labels[:5])
        soft_label = mu / mu.sum( axis= 1 ).reshape(-1,1).repeat( axis=1, repeats=self.Ndom)
        mu = ( mu.max(axis=1).reshape(-1,1).repeat(axis=1,repeats=self.Ndom ) == mu)
        mu = mu.astype(float)
        mu = mu / mu.sum(axis=1).reshape(-1,1).repeat(axis=1,repeats=self.Ndom)
        tmp1 = np.array(range(1,1 + self.Ndom)).reshape(1,-1).repeat(axis=0,repeats = true_labels.shape[0] )
        tmpTrue = true_labels.reshape(-1,1).repeat(axis=1, repeats=self.Ndom)
        error_rate = ( ( tmpTrue != tmp1 ) * mu ).sum(axis=1).mean()
        soft_error_rate = ( ( tmpTrue != tmp1 ) * soft_label).sum(axis=1).mean()
        return error_rate, soft_error_rate, -1, -1

if __name__ == "__main__":
    mat = sio.loadmat("web_ans_soft.mat")
    ans_soft = mat["ans_soft_labels"]
    D = CrowdsourcingModel()
    D.loadData('./datasets/mat/web_data.mat')
    print(D.cal_error_using_soft_label(ans_soft, D.true_labels))