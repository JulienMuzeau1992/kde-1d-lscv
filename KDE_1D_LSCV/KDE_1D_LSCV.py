import numpy as np
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
from tqdm import tqdm

class KDE_1D_LSCV:
    def __init__(self, kernel="gaussian"):
        self.kernel = kernel
        self.K, self.K2, self.R_K = self.kernel_functions(self.kernel)
        self.X = None
        self.h = None

    def kernel_functions(self, kernel):
        if kernel == "gaussian":
            # Kernel function
            def G(u):
                return 1/np.sqrt(2*np.pi) * np.exp(-1/2 * u**2)

            # Kernel auto-convolution
            def G2(u):
                return 1/(2*np.sqrt(np.pi)) * np.exp(-1/4 * u**2)

            # 2nd-order moment
            R_G = 1/(2 * np.sqrt(np.pi))

            return G, G2, R_G
        else:
            raise ValueError("Only 'gaussian' kernel is supported")
    
    def fit(self, X, bw_range, n_jobs=1):
        if len(X.shape) != 2 or X.shape[1] != 1:
            raise ValueError("'X' must be (n_samples, 1)")
        
        self.X = X
        self.pairwise_distances = pairwise_distances(self.X)

        self.h = self.compute_LSCV_bandwidth(
            bw_range=bw_range,
            n_jobs=n_jobs)

    def compute_LSCV_bandwidth(self, bw_range, n_jobs=1):
        scores = Parallel(n_jobs=n_jobs)(delayed(self.LSCV)(bw) for bw in tqdm(bw_range))
        return bw_range[np.argmin(scores)]

    def LSCV(self, h):
        n = len(self.X)
        lscv = self.R_K
        lscv += 2/n * np.sum(np.triu(self.K2(self.pairwise_distances/h), k=1))
        lscv -= 4/(n-1) * np.sum(np.triu(self.K(self.pairwise_distances/h), k=1))
        lscv /= (n * h)
        return lscv

    def estimate(self, x):
        return 1/self.h * np.mean(self.K((x - self.X)/self.h), axis=0)