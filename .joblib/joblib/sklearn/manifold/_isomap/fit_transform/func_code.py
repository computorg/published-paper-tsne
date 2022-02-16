# first line: 305
    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse graph, BallTree, KDTree}
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            X transformed in the new space.
        """
        self._fit_transform(X)
        return self.embedding_
