# first line: 751
    def fit_transform(self, X, y=None):
        """Compute the embedding vectors for data X and transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training set.

        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Returns the instance itself.
        """
        self._fit_transform(X)
        return self.embedding_
