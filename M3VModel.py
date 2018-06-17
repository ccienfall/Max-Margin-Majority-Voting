import numpy as np
import scipy as sp
import scipy.io as sio
from crowdsourcingModel import CrowdsourcingModel

class M3VModel(CrowdsourcingModel):
    def initial(self):
        K = self.Ndom
        Ndom = K
        self.A0 = np.zeros((1,self.Nwork))
        self.B0 = np.mat(np.diag([1/float(self.v) for i in range(self.Nwork)]))
        self.probd0 = np.ones( (Ndom, K, self.Nwork) )
        self.eta = np.dot( self.A0, self.B0.I)
        self.phi = np.zeros((Ndom, K, self.Nwork))
        self.ilm = np.ones(( self.Ntask, 1))
        self.Y = np.zeros( ( self.Ntask, 1)).astype(int)
        self.S = np.zeros( ( self.Ntask, 1)).astype(int)
        self.ans_soft_labels = np.zeros(( self.Ntask, K) )
        self.phic = np.zeros( ( Ndom, K, self.Nwork))
        self.etak = np.zeros( ( self.maxIter - self.burnIn, self.Nwork))
        self.etak_count = 0
        self.X = np.zeros( (self.Ntask, K, self.Nwork))
        for i in range(self.Ntask):
            for j in self.NeibTask[i]:
                self.X[i, self.L[i,j] - 1, j] = 1
        None

    def initByMajorityVoting(self):
        if not hasattr(self,'Y'):
            self.Y = np.zeros((self.Ntask, 1))
        for i in range(self.Ntask):
            ct = np.zeros((self.Ndom, 1))
            for j in range(len(ct)):
                ct[j] = sum( self.L[i,:] == j+1 )
            self.Y[i] = ct.argmax() + 1
        None