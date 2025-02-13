import numpy as np

from sklearn.mixture import GaussianMixture

class GMmodel:
    def __init__(self, n_iter, tol, covariance_type, n_component=None):
        self.n_component = n_component
        self.n_iter = n_iter
        self.tol = tol
        self.covariance_type = covariance_type

    def gmm(self, input_data):
        if self.n_component:
            gmm = GaussianMixture(
                n_components=self.n_component,
                max_iter=self.n_iter,
                tol=self.tol,
                covariance_type=self.covariance_type
            )
        else:
            self.n_component = self._auto_components(input_data)
            gmm = GaussianMixture(
                n_components=self.n_component,
                max_iter=self.n_iter,
                tol=self.tol,
                covariance_type=self.covariance_type
            )
        return gmm.fit(input_data)

    def _auto_components(self, input_data, max_components=10):
        """
        Determine the optimal number of components using BIC.

        Args:
            input_data (array-like): The data to fit the GMM.
            max_components (int): The maximum number of components to test.

        Returns:
            int: The optimal number of components based on the lowest BIC.
        """
        lowest_bic = np.inf
        best_n_components = 1

        for _n_components in range(1, max_components + 1):
            gmm = GaussianMixture(
                n_components=_n_components,
                max_iter=self.n_iter,
                tol=self.tol,
                covariance_type=self.covariance_type
            )
            gmm.fit(input_data)
            bic = gmm.bic(input_data)
            if bic < lowest_bic:
                lowest_bic = bic
                best_n_components = _n_components

        self.n_component = best_n_components
        return best_n_components

if __name__ == "__main__":
    # Example usage
    # Generate sample data
    np.random.seed(0)
    C = np.array([[0.0, -0.1], [1.7, 0.4]])
    component_1 = np.dot(np.random.randn(500, 2), C)
    component_2 = 0.7 * np.random.randn(500, 2) + np.array([-4, 1])
    X = np.vstack([component_1, component_2])

    # Initialize GMmodel without specifying n_component to enable auto-selection
    gmm_model = GMmodel(n_iter=100, tol=1e-3, covariance_type='full')

    # Fit the model to the data
    fitted_gmm = gmm_model.gmm(X)

    # Output the optimal number of components
    print(f"Optimal number of components: {gmm_model.n_component}")
    
    # Plot the original data and the GMM
    import matplotlib.pyplot as plt
    
    # Sample data from the GMM
    _samples, _ = fitted_gmm.sample(1000)
    
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.scatter(_samples[:, 0], _samples[:, 1], marker='x', color='red')
    plt.scatter(X[:, 0], X[:, 1], s=1, c = 'blue')
    plt.savefig('data_augmentation/GMM/alg/gmm.png')
