import numpy as np

from csb.numeric import log_sum_exp
from csb.bio.utils import wfit, rmsd, fit_wellordered, distance_matrix, average_structure
from csb.bio.io.wwpdb import RemoteStructureProvider as PDB
from csb.statistics import principal_coordinates

from em import EMFitter, SegmentMixture, SegmentMixture2, Reporter

from csb.statistics.rand import random_rotation
from csb.numeric import log_sum_exp, log

def load_coordinates(codes):
    '''
    Load protein 3 dimensional coordinate structure based on Carbon alpha.

    :param codes: PDB Code and sid chain (eg: 1AKE_A)
    :return:
    '''
    atoms = ['CA']
    X = []
    for code in codes:
        name, chainid = code.split('_')
        struct = PDB().get(name)
        chain = struct[chainid]
        X.append([residue[atom].vector for residue in chain for atom in atoms])

    return np.array(X)

class GibbsSampler(object):

    def __init__(self, X, K, estimate_sigma=True, prior=1):

        super(GibbsSampler, self).__init__(X, K, estimate_sigma)

        self.sigma_t = 1e1
        self.sigma_Y = 1e1
        self.alpha_precision = 0.1
        self.beta_precision = 0.1

        self._prior = int(prior)

        if not self.sequential_prior:
            self.alpha_w[:] = 1. / K
        else:
            self.alpha_w = 0.95
            self.w = self.alpha_w

    @property
    def sequential_prior(self):
        return self._prior == 2
        
    def estimate_w(self):

        if not self.sequential_prior:

            N = self.Z.sum(0)
            self.w[:] = np.random.dirichlet(N + self.alpha_w, 1)

        else:

            l = self.membership
            Q = np.sum(l[1:] == l[:-1])

            self.w = np.random.beta(Q + self.alpha_w, self.N - Q - self.alpha_w)

    def random_membership(self):

        if not self.sequential_prior:

            super(GibbsSampler, self).random_membership()

        else:

            l = np.zeros(self.N,'i')
            l[0] = np.random.randint(0, self.K+1)
            for n in range(1,len(l)):
                if np.random.random() < self.w:
                    l[n] = l[n-1]
                else:
                    l[n] = np.random.choice(list(set(range(self.K))-set([l[n-1]])))

            self.membership = l

    def estimate_sigma(self, beta=1.):

        N    = self.D * self.M * self.Z.sum(0)
        chi2 = np.sum(self.Z * self.delta, 0)
        
        a = 0.5 * beta * N    + self.alpha_precision
        b = 0.5 * beta * chi2 + self.beta_precision

        self.sigma[:] = (b / np.random.gamma(a))**0.5

    def estimate_Z(self, beta=1.):

        prob  = - 0.5 * beta * self.delta / self.sigma**2 \
                - self.D * beta * self.M * np.log(self.sigma) 

        if not self.sequential_prior:

            prob += log(self.w)
            prob  = np.exp((prob.T - log_sum_exp(prob.T, 0)).T)

            for n in range(self.N):
                self.Z[n,:] = np.random.multinomial(1, prob[n])

        else:

            a = log(self.w)
            b = log((1-self.w)/(self.K-1))
            
            for n in range(self.N):

                p = prob[n]
                if n > 1:
                    p += self.Z[n-1] * a + (1-self.Z[n-1]) * b

                p = np.exp(p - log_sum_exp(p))
            
                self.Z[n,:] = np.random.multinomial(1, p)

    def estimate_R(self, beta=1.):

        y = np.sum(self.Y.swapaxes(0,2) * self.Z, 1).T

        for m in range(self.M):
            for k in range(self.K):

                A = np.dot(self.Y[k].T * self.Z[:,k], self.X[m]) - \
                    np.outer(y[k], self.t[m,k])
                A*= beta / self.sigma[k]**2

                self.R[m,k] = random_rotation(A.T)

    def estimate_t(self, beta=1.):

        N = self.Z.sum(0)

        for m in range(self.M):
            for k in range(self.K):

                mu  = np.dot(self.Z[:,k], self.X[m] - np.dot(self.Y[k], self.R[m,k].T))
                mu *= beta / self.sigma[k]**2

                prec = beta * N[k] / self.sigma[k]**2 + 1. / self.sigma_t**2
                sigma = 1 / prec**0.5
                mu /= prec

                self.t[m,k] = np.random.standard_normal(self.D) * sigma + mu

    def estimate_T(self, beta=1.):

        self.estimate_R(beta)
        self.estimate_t(beta)

    def e_step(self, beta=1.):
        self.estimate_Z(beta=1.)

    def estimate_Y(self, beta=1.):

        for k in range(self.K):

            mu = 0.
            for m in range(self.M):
                mu += np.dot(self.X[m]-self.t[m,k], self.R[m,k])
                
            prec  = self.M * beta / self.sigma[k]**2 + 1. / self.sigma_Y**2
            mu   *= (beta / self.sigma[k]**2) / prec
            sigma = 1 / prec**0.5 

            mu    = mu.T #* self.Z[:,k]
            #sigma = sigma * self.Z[:,k] + self.sigma_Y * (1-self.Z[:,0])
            
            self.Y[k,...] = (np.random.standard_normal(mu.shape) * sigma + mu).T

    def m_step(self, beta=1.):

        self.estimate_T(beta)
        self.estimate_Y(beta)
        self.estimate_w()

        if self._estimate_sigma:
            self.estimate_sigma(beta)
        
    def anneal(self, beta, initialize=True, n_iter=1, verbose=False):

        if initialize: self.initialize()

        self.L = []

        for i in range(len(beta)):

            L = []

            for j in range(n_iter):

                self.m_step(beta[i])
                self.e_step(beta[i])
                self.invalidate_delta()

                L.append(self.log_likelihood)

            self.L.append(np.mean(L))

            if verbose and not i % verbose:
                print beta[i], self.L[-1]
            
    def anneal_prior(self, alpha, initialize=True, n_iter=1, verbose=False):
        """
        Inspired by Mengersen et al. 
        """
        if initialize: self.initialize()

        self.L = []

        for i in range(len(alpha)):

            L = []
            self.alpha_w[:] = alpha[i]
            
            for j in range(n_iter):

                self.m_step()
                self.e_step()
                self.invalidate_delta()

                L.append(self.log_likelihood)

            ## store log likelihood and entropy
                
            self.L.append((np.mean(L), np.sum(self.Z.sum(0)>0), -np.sum(self.w * np.log(self.w+1e-300))))

            if verbose and not i % verbose:
                print alpha[i], self.L[-1], self.w.min()
            
    @property
    def log_likelihood(self):
        """
        returns the log-likelihood
        """
        N = self.Z.sum(0)

        ## TODO: there is a bug in the CSB implementation
        
        L = - 0.5 * np.sum(self.Z * self.delta / self.sigma**2) \
            - 0.5 * self.D * self.M * np.sum(N * log(2 * np.pi * self.sigma**2))

        return L

    @property
    def log_prior(self):

        N = self.Z.sum(0)

        p = -0.5 * np.sum(self.t**2) / self.sigma_t**2
        p+= -0.5 * np.sum(self.Y**2) / self.sigma_Y**2
        p+= np.sum((self.alpha_precision-1) * log(1/self.sigma**2) - self.beta_precision / self.sigma**2)
        
        if not self.sequential_prior:
            p += np.sum((N + self.alpha_w - 1) * np.log(self.w))

        else:
            l = self.membership
            Q = np.sum(l[1:] == l[:-1])
            p += (Q + self.alpha_w - 1) * log(self.w) + \
                 (self.N - Q - self.alpha_w - 1) * log(1-self.w)

        return p

    @property
    def log_posterior(self):
        return self.log_likelihood + self.log_prior

class GibbsSampler2(GibbsSampler):

    def __init__(self, X, K, prior=1):
        
        self._Y = np.zeros((X.shape[1], self.D))
        super(GibbsSampler2, self).__init__(X, K, prior=prior)

    @property
    def Y(self):
        return np.array([self._Y] * self.K)

    @Y.setter
    def Y(self, Y):
        if Y.ndim == 2:
            self._Y[:,:] = Y[:,:]
        elif Y.ndim == 3:
            self.Y = Y[0]

    def estimate_Y(self, beta=1.):

        Y = 0.
        for k in range(self.K):

            mu = 0.
            for m in range(self.M):
                mu += np.dot(self.X[m]-self.t[m,k], self.R[m,k])
            Y += mu.T * self.Z[:,k] / self.sigma[k]**2

        prec  = self.M * beta * np.dot(self.Z, 1 / self.sigma**2) + 1. / self.sigma_Y**2
        Y    *= beta / prec
        sigma = 1 / prec**0.5 
            
        self._Y[...] = (np.random.standard_normal(Y.shape) * sigma + Y).T

    def estimate_Y(self, beta=1.):

        self._Y[...] = 0.

        for k in range(self.K):

            mu = 0.
            for m in range(self.M):
                mu += np.dot(self.X[m]-self.t[m,k], self.R[m,k])
                
            prec  = self.M * beta / self.sigma[k]**2 + 1. / self.sigma_Y**2
            mu   *= (beta / self.sigma[k]**2) / prec
            sigma = 1 / prec**0.5 
            
            self._Y += ((np.random.standard_normal(mu.shape) * sigma + mu).T * self.Z[:,k]).T

class _GibbsSampler2(SegmentMixture2):

    def __init__(self, X, K, prior=1):

        super(GibbsSampler2, self).__init__(X, K)

        self.sigma_t = 1e1
        self.sigma_Y = 1e1
        self.alpha_precision = 0.1
        self.beta_precision = 0.1
        
        self._prior = int(prior)

        if not self.sequential_prior:
            self.alpha_w[:] = 1. / K
        else:
            self.alpha_w = 0.95
            self.w = self.alpha_w

    @property
    def sequential_prior(self):
        return self._prior == 2
        
    def estimate_w(self):

        if not self.sequential_prior:

            N = self.Z.sum(0)
            self.w[:] = np.random.dirichlet(N + self.alpha_w, 1)

        else:

            l = self.membership
            Q = np.sum(l[1:] == l[:-1])

            self.w = np.random.beta(Q + self.alpha_w, self.N - Q - self.alpha_w)

    def random_membership(self):

        if not self.sequential_prior:

            super(GibbsSampler2, self).random_membership()

        else:

            l = np.zeros(self.N,'i')
            l[0] = np.random.randint(0, self.K+1)
            for n in range(1,len(l)):
                if np.random.random() < self.w:
                    l[n] = l[n-1]
                else:
                    l[n] = np.random.choice(list(set(range(self.K))-set([l[n-1]])))

            self.membership = l

    def estimate_sigma(self, beta=1.):

        N    = self.D * self.M * self.Z.sum(0)
        chi2 = np.sum(self.Z * self.delta, 0)
        
        a = 0.5 * beta * N    + self.alpha_precision
        b = 0.5 * beta * chi2 + self.beta_precision

        self.sigma[:] = (b / np.random.gamma(a))**0.5

    def estimate_Z(self, beta=1.):

        prob  = - 0.5 * beta * self.delta / self.sigma**2 \
                - self.D * beta * self.M * np.log(self.sigma) 

        if not self.sequential_prior:

            prob += log(self.w)
            prob  = np.exp((prob.T - log_sum_exp(prob.T, 0)).T)

            for n in range(self.N):
                self.Z[n,:] = np.random.multinomial(1, prob[n])

        else:

            a = log(self.w)
            b = log((1-self.w)/(self.K-1))
            
            for n in range(self.N):

                p = prob[n]
                if n > 1:
                    p += self.Z[n-1] * a + (1-self.Z[n-1]) * b

                p = np.exp(p - log_sum_exp(p))
            
                self.Z[n,:] = np.random.multinomial(1, p)

    def estimate_R(self, beta=1.):

        y = np.dot(self.Z.T, self.Y)

        for m in range(self.M):
            for k in range(self.K):

                A = np.dot(self.Y.T * self.Z[:,k], self.X[m]) - \
                    np.outer(y[k], self.t[m,k])
                A*= beta / self.sigma[k]**2

                self.R[m,k] = random_rotation(A.T)

    def estimate_t(self, beta=1.):

        N = self.Z.sum(0)

        for m in range(self.M):
            for k in range(self.K):

                mu = np.dot(self.Z[:,k], self.X[m] - np.dot(self.Y, self.R[m,k].T))
                mu = beta * mu / self.sigma[k]**2

                prec = beta * N[k] / self.sigma[k]**2 + 1. / self.sigma_t**2
                sigma = 1 / prec**0.5
                mu /= prec

                self.t[m, k] = np.random.standard_normal(self.D) * sigma + mu

    def estimate_T(self, beta=1.):

        self.estimate_R(beta)
        self.estimate_t(beta)

    def e_step(self, beta=1.):
        self.estimate_Z(beta=1.)

    def estimate_Y(self, beta=1.):

        Y = 0.
        for k in range(self.K):

            mu = 0.
            for m in range(self.M):
                mu += np.dot(self.X[m]-self.t[m,k], self.R[m,k])
            Y += mu.T * self.Z[:,k] / self.sigma[k]**2

        prec  = self.M * beta * np.dot(self.Z, 1 / self.sigma**2) + 1. / self.sigma_Y**2
        Y    *= beta / prec
        sigma = 1 / prec**0.5 
            
        self._Y[...] = (np.random.standard_normal(Y.shape) * sigma + Y).T

    def m_step(self, beta=1.):

        self.estimate_T(beta)
        self.estimate_Y(beta)
        self.estimate_w()
        self.estimate_sigma(beta)
        
    def anneal(self, beta, initialize=True, n_iter=1, verbose=False):

        if initialize: self.initialize()

        self.L = []

        for i in range(len(beta)):

            L = []

            for j in range(n_iter):

                self.m_step(beta[i])
                self.e_step(beta[i])
                self.invalidate_delta()
                L.append(self.log_likelihood)

            self.L.append(np.mean(L))

            if verbose and not verbose % i:
                print beta[i], self.L[-1]
            
    @property
    def log_likelihood(self):
        """
        returns the log-likelihood
        """
        N = self.Z.sum(0)
        L = - 0.5 * np.sum(self.Z * self.delta / self.sigma**2) \
            - 0.5 * self.D * self.M * np.sum(N * log(2 * np.pi * self.sigma**2))

        return L

    @property
    def log_prior(self):

        N = self.Z.sum(0)

        p = -0.5 * np.sum(self.t**2) / self.sigma_t**2
        p+= -0.5 * np.sum(self.Y**2) / self.sigma_Y**2
        p+= np.sum((self.alpha_precision-1) * log(1/self.sigma**2) - self.beta_precision / self.sigma**2)
        
        if not self.sequential_prior:
            p += np.sum((N + self.alpha_w - 1) * np.log(self.w))

        else:
            l = self.membership
            Q = np.sum(l[1:] == l[:-1])
            p += (Q + self.alpha_w - 1) * log(self.w) + \
                 (self.N - Q - self.alpha_w - 1) * log(1-self.w)

        return p

    @property
    def log_posterior(self):
        return self.log_likelihood + self.log_prior

if __name__ == '__main__':

    Y = X.copy()
    for m in range(len(Y)):
        Y[m] -= Y[m].mean(0)

    K = 10 #20
    gibbs = GibbsSampler(Y, K, prior=2)
    info = Reporter()
    info.info['w'] = []
    info.info['membership'] = []
    #if gibbs.sequential_prior: gibbs.w = 0.75
    gibbs.run(200,reporter=info)

    print gibbs.log_likelihood

    B = np.sum([np.equal.outer(m,m) for m in info.membership[100:]],0)

    info = Reporter()
    info.info['w'] = []
    info.info['membership'] = []
    gibbs2 = GibbsSampler2(Y, K, prior=1)
    gibbs2.initialize()
    gibbs2.run(200,initialize=False,reporter=info)

    print gibbs2.log_likelihood

    B2 = np.sum([np.equal.outer(m,m) for m in info.membership[100:]],0)







    
