import numpy as np
from KDE_1D_LSCV.KDE_1D_LSCV import KDE_1D_LSCV
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Generate random data
    data = np.concatenate([
        np.random.normal(loc=0, size=100),
        np.random.normal(loc=4, size=200)
    ]).reshape(-1, 1)

    # Kernel density estimation with LSCV
    kde = KDE_1D_LSCV()
    kde.fit(data, bw_range=np.logspace(-2, 1, 1000), n_jobs=-1)
    print("Optimal bandwidth according to LSCV = ", kde.h)

    # Plot
    l = np.min(data)
    u = np.max(data)
    rng = u - l
    x = np.linspace(np.floor(l - 0.5*rng),
                    np.ceil(u + 0.5*rng),
                    200)
    y = kde.estimate(x)

    plt.figure(1)
    plt.title("Univariate kernel density estimation with Least Squares Cross-Validation")
    plt.plot(x, y, label="Estimated pdf")
    plt.hist(data, bins=50, density=True, color="r", alpha=0.5, label="Normalized histogram")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

