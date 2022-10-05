"""
A Python implementation of the evidential k nearest neighbours for imperfectly labeled data.
EK-NN was first introduced by T. Denoeux and this version is based on a source code developed by Daniel Zhu.

Author : Arthur Hoarau
Date : 26/10/2021
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score
from lib import ibelief
import numpy as np
import math

# Value of the Alpha parameter
ALPHA = 0.8
BETA = 2

class EKNN(BaseEstimator, ClassifierMixin):
    """
    EK-NN class used to predict labels when input data 
    are imperfectly labeled.
    
    Based on the Evidental k nearest neighbours (EKNN) classifier by Denoeux (1995).
    """

    def __init__(self, class_number, n_neighbors=5):
        """
        EK-NN class used to predict labels when input data 
        are imperfectly labeled.

        Parameters
        -----
        class_number: int
            The number of classes for the problem. Dimension of the possible classes.
        n_neighbors : int
            Number of nearest neighbors, default = 5

        Returns
        -----
        The instance of the class.
        """

        # Used to retrieve the n nearest neighbors
        self.n_neighbors = n_neighbors

        # Select number of classes
        self.nb_classes = 2**class_number - 1 

        # Used to retrieve the state of the model
        self._fitted = False

    def get_params(self):
        # Return the number of nearest neighbors as a dict
        return {"n_neighbours": self.n_neighbors}

    def set_params(self, n_neighbors):
        # Set the number of nearest neighbors
        self.n_neighbors = n_neighbors

    def score(self, X, y_true, criterion=3):
        """
        Calculate the accuracy score of the model,
        unsig a specific criterion in "Max Credibility", 
        "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X's
        y_true : ndarray
            True labels of X, to be compared with the model predictions.
        criterion : int
            Choosen criterion for prediction, by default criterion = 3.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".

        Returns
        -----
        The accuracy score of the model.
        """

        # Make predictions on X, using the given criterion
        y_pred = self.predict(X, criterion=criterion)

        # Compare with true labels, and compute accuracy
        return accuracy_score(y_true, y_pred)
    
    def fit(self, X, y, alpha=ALPHA, beta=BETA, unique_gamma=False):
        """
        Fit the model according to the training data.

        Parameters
        -----
        X : ndarray
            Input array of X's
        y : ndarray
            Labels array
        alpha : int
            Value of the alpha parameter, default = 0.95
        beta : int
            Value of the beta parameter, default = 1.5
        unique_gamma : boolean
            True for a unique computation of a global gamma parameter, 
            False for multiple gammas (high computational cost). default = True.
        Returns
        -----
        self : EKNN
            The instance of the class.
        """

        # Check for data integrity
        if X.shape[0] != y.shape[0]:
            if X.shape[0] * (self.nb_classes + 1) == y.shape[0]:
                y = np.reshape(y, (-1, self.nb_classes + 1))
            else:
                raise ValueError("X and y must have the same number of rows")

        # Verify if the size of y is of a power set (and if it contains the empty set or not)
        if math.log(y.shape[1], 2).is_integer():
            y = y[:,1:]
        elif not math.log(y.shape[1] + 1, 2).is_integer():
            raise ValueError("y size must be the size of the power set of the frame of discernment")

        # Save X and y
        self.X_trained = X
        self.y_trained = y

        # Save size of the dataset
        self.size = self.X_trained.shape[0]

        # Init gamma and alpha
        self._init_parameters(alpha=alpha, unique_gamma=unique_gamma, beta=beta)

        # The model is now fitted
        self._fitted = True

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled

        Returns
        -----
        predictions : ndarray
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        result = self._predict(X)

        predictions = ibelief.decisionDST(result.T, 4, return_prob=True)

        return predictions

    def predict(self, X, criterion=3, return_bba=False):
        """
        Predict labels of input data. Can return all bbas. Criterion are :
        "Max Credibility", "Max Plausibility" and "Max Pignistic Probability".

        Parameters
        -----
        X : ndarray
            Input array of X to be labeled
        creterion : int
            Choosen criterion for prediction, by default criterion = 1.
            1 : "Max Plausibility", 2 : "Max Credibility", 3 : "Max Pignistic Probability".
        return_bba : boolean
            Type of return, predictions or both predictions and bbas, 
            by default return_bba=False.

        Returns
        -----
        predictions : ndarray
        result : ndarray
            Predictions if return_bba is False and both predictions and masses if return_bba is True
        """

        # Verify if the model is fitted or not
        if not self._fitted:
            raise NotFittedError("The classifier hasn not been fitted yet")

        # Predict output bbas for X
        result = self._predict(X)

        # Max Plausibility
        if criterion == 1:
            predictions = ibelief.decisionDST(result.T, 1)
        # Max Credibility
        elif criterion == 2:
            predictions = ibelief.decisionDST(result.T, 2)
        # Max Pignistic probability
        elif criterion == 3:
            predictions = ibelief.decisionDST(result.T, 4)
        else:
            raise ValueError("Unknown decision criterion")

        # Return predictions or both predictions and bbas
        if return_bba:
            return predictions, result
        else:
            return predictions

    def _compute_bba(self, X, indices, distances):
        """
        Compute the bba for each element of X.

        Parameters
        -----
        X : ndarray
            Input array of X
        indices : ndarray
            Array of K nearest neighbors indices
        distances : ndarray
            Array of K nearest neighbors distances

        Returns
        -----
        bba : ndarray
            Array of bbas
        """
        # Initialisation of size and all bba
        n_samples = X.shape[0]
        bba = np.zeros((n_samples, self.nb_classes + 1))

        # Calculate a bba for each element of X
        for i in range(n_samples):
            m_list = np.zeros((self.n_neighbors, self.nb_classes + 1))

            # Construct a bba for each neighbors
            for j in range(self.n_neighbors):
                m = np.zeros(self.nb_classes + 1)
                m[-1] = 1

                for c in range(m.shape[0] - 2):
                    if isinstance(self.gamma, float):
                        weight = self.alpha * math.exp((-self.gamma) * (distances[i,j] ** self.beta)) * self.y_trained[int(indices[i,j]), c]
                    else:
                        weight = self.alpha * math.exp((-self.gamma[int(indices[i,j])]) * (distances[i,j] ** self.beta)) * self.y_trained[int(indices[i,j]), c]
                    m[c + 1] = weight
                    m[-1] -= weight

                m_list[j] = m
            
            # Compute normalized combination of bba
            m_normalized = np.array(ibelief.DST(m_list.T, 2))

            # Append the normalized bba to the array
            bba[i] = m_normalized.T
        
        return bba

    def _compute_distances(self, X):
        """
        Compute the euclidian distances with each neighbors.

        Parameters
        -----
        X : ndarray
            Input array of X

        Returns
        -----
        indices : ndarray
            Array of K nearest neighbors indices
        distances : ndarray
            Array of K nearest neighbors distances
        """

        # Initialize indices and nearest neighbors and distances
        indices = np.zeros((X.shape[0], self.n_neighbors))
        distances = np.zeros((X.shape[0], self.n_neighbors))

        # Loop over every input sample
        for i in range(X.shape[0]):

            # Compute the distance (without sqrt)
            dist = np.sqrt(np.sum(([X[i]] - self.X_trained)**2,axis=1))
            sorted_indices = np.argsort(dist)[:self.n_neighbors]

            # Append result to each arrays
            indices[i] = sorted_indices
            distances[i] = dist[sorted_indices]
        return indices, distances

    def _predict(self, X):
        """
        Compute distances and predicted bba on the input.

        Parameters
        -----
        X : ndarray
            Input array of X

        Returns
        -----
        result : ndarray
            Array of normalized bba
        """
        
        # Compute distances with k nearest neighbors
        neighbours_indices, neighbours_distances = self._compute_distances(X)

        # Compute bba
        result = self._compute_bba(X, neighbours_indices, neighbours_distances)

        return result

    def _init_parameters(self, alpha=ALPHA, beta=BETA, unique_gamma=False):
        # Init alpha and beta
        self.alpha = alpha
        self.beta = beta

        # Init parameter gamma
        self.gamma = self._compute_gamma(unique_gamma=unique_gamma)

    def _compute_gamma(self, unique_gamma=False):
        """
        Compute gamma parameter. Either unique or multiple.

        Returns
        -----
        gamma : ndarray
            Array of gamma parameters
        or
        gamma : int
            Value of gamma
        """

        if(unique_gamma):
            # Initialize distances and divider term
            divider = (self.size**2 - self.size) if self.size > 1 else 1
            distances = np.zeros((self.size, self.size))

            # Compute euclidian distances between each point
            for i, x in enumerate(self.X_trained):
                distances[i] = np.sqrt(
                    np.sum(([x] - self.X_trained)**2,axis=1)
                )

            mean_distance = np.sum(distances) / divider
            return 1 / (mean_distance  ** self.beta)

        # Initialize distances and divider term
        gamma = np.zeros(self.size)

        jousselme_distance = np.zeros((self.size, self.size))
        norm_distances = np.zeros((self.size, self.size))

        # Compute Jousselme and norm distances
        for i in range(self.size):

            # Init masses
            bbai = np.zeros(self.nb_classes + 1)
            bbai[1:] = self.y_trained[i]

            D = ibelief.Dcalculus(np.array(bbai).reshape((1,bbai.size)).size)

            for j in range(self.size):
                
                # Init masses
                bbaj = np.zeros(self.nb_classes + 1)
                bbaj[1:] = self.y_trained[j]

                jousselme_distance[i, j] = ibelief.JousselmeDistance(bbai, bbaj, D)

        norm_distances = np.array([[np.linalg.norm(i-j) for j in self.X_trained] for i in self.X_trained])

        for n in range(self.size):

            # Init the bba
            bban = np.zeros(self.nb_classes + 1)
            bban[1:] = self.y_trained[n]

            jousselm_distances_matrix = np.zeros((1, self.size))
            jousselm_distances_matrix[0] =  1 - jousselme_distance[n]

            jousselm_product = np.matmul(jousselm_distances_matrix.T, jousselm_distances_matrix)

            # Buffer not to compute multiple times the operation
            dividend = jousselm_product * norm_distances
            divisor = np.sum(jousselm_product) - np.sum(np.diagonal(jousselm_product))

            # If Jousselme distances are nulls
            if divisor == 0:
                divisor = 1

            gamma[n] = 1 / ((np.sum(dividend) / divisor) ** self.beta)
        return gamma
