import numpy as np
from csb.numeric import log_sum_exp
from csb.bio.utils import wfit, average_structure

class SegmentMixture(object):
    """
    Faster implementation than the one in CSB
    """
    D = 3

    def __init__(self, X, K, estimate_sigma=True):

        self.X = X
        self.K = int(K)

        self.R = np.zeros((self.M, self.K, self.D, self.D))
        self.t = np.zeros((self.M, self.K, self.D))
        self.Y = np.zeros((self.K, self.N, self.D))
        self.sigma = np.ones(self.K)
        self.w = np.ones(self.K) / self.K
        self.Z = np.zeros((self.N, self.K))

        self._delta = None

        ## hyperparameters

        self.alpha_precision = 1e-2
        self.beta_precision  = 1e-2

        self.alpha_w = 0.01 * np.ones(self.K) / self.K

        self._estimate_sigma = estimate_sigma
        if not estimate_sigma:
            ## assumes that intra segment rmsd is 1
            self.sigma[:] = 3.**(-0.5)
        
    @property
    def N(self):
        return self.X.shape[1]

    @property
    def M(self):
        return self.X.shape[0]

    def initialize(self):

        R = np.kron(np.ones((self.M, self.K)), np.eye(self.D))
        Y = average_structure(self.X)
        
        self.Y = np.array([Y.copy() for _ in range(self.K)])
        self.R[...] = R.reshape(self.M,self.K,self.D,self.D)

        self.random_membership()

    def random_membership(self):

        alpha = np.ones(self.K)

        self.Z[...] = np.random.dirichlet(alpha, self.N)
        self.w[...] = self.Z.sum(0) / self.N
        self.Z[...] = [np.random.multinomial(1, self.w) for _ in range(self.N)]

    @property
    def delta(self):

        if self._delta is None:

            d = np.zeros((self.M, self.K, self.N))

            for k in range(self.K):
                for m in range(self.M):
                    X = np.dot(self.Y[k],self.R[m,k].T) + self.t[m,k]
                    d[m,k] = np.sum((self.X[m] - X)**2, -1)

            self._delta = d.sum(0).T

        return self._delta

    def invalidate_delta(self):
        self._delta = None

    def e_step(self):

        p = - 0.5 * self.delta / self.sigma**2 - 3 * self.M * np.log(self.sigma) + np.log(self.w)
        p = np.exp((p.T - log_sum_exp(p.T, 0)).T)

        self.Z[:,:] = p

    def estimate_T(self):

        for m in range(self.M):
            for k in range(self.K):
                self.R[m,k], self.t[m, k] = wfit(self.X[m], self.Y[k], self.Z[:, k])

    def estimate_Y(self):

        self.Y[:,:,:] = 0.
        for k in range(self.K):
            for m in range(self.M):
                self.Y[k,:,:] += np.dot(self.X[m] - self.t[m,k], self.R[m,k]) / self.M

    def estimate_w(self):
        
        N = self.Z.sum(0)

        self.w[:] = (N + self.alpha_w) / (self.N + self.alpha_w.sum())

    def estimate_sigma(self):

        N = self.Z.sum(0)

        a = 1.5 * self.M * N + self.alpha_precision
        b = 0.5 * np.sum(self.Z * self.delta, 0) + self.beta_precision

        self.sigma[:] = (b/a)**0.5

    def m_step(self):

        self.estimate_T()
        self.estimate_Y()
        self.estimate_w()
        self.invalidate_delta()
        if self._estimate_sigma:
            self.estimate_sigma()
        
    def run(self, niter=20, initialize=True, reporter=None):

        if initialize: self.initialize()

        for i in range(niter):

            self.m_step()
            self.e_step()
            self.invalidate_delta()

            if reporter is not None:
                reporter.update(self)
            
    @property
    def membership(self):
        return self.Z.argmax(1)

    @membership.setter
    def membership(self, m):

        self.Z[:,:] = np.equal.outer(m, np.arange(self.K))

    @property
    def log_likelihood(self):
        """
        returns the log-likelihood
        """
        N = self.Z.sum(0)

        ## TODO: there is a bug in the CSB implementation
        
        L = - 0.5 * np.sum(self.Z * self.delta / self.sigma**2) \
            - 0.5 * self.D * self.M * np.sum(N * np.log(2 * np.pi * self.sigma)) \
            + np.sum(N * np.log(self.w))
            
        return L

    @property
    def log_likelihood(self):
        """
        returns the log-likelihood
        """
        ## This is the log likelihood if we don't anneal w_k
        
        N = self.Z.sum(0)
        L = - 0.5 * np.sum(self.Z * self.delta / self.sigma**2) \
            - 0.5 * self.D * self.M * np.sum(N * np.log(2 * np.pi * self.sigma))
            
        return L


