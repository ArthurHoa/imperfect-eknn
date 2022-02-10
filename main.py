from eknn_imperfect import EKNN
import numpy as np

"""
Example with 2 imperfectly labeled observations.
"""

# Training X values [x1, x2]
X_train = np.array([[0.5, 0.5],
                    [-0.5, -0.5]])

# Training y values, basic belief assigments (with or without empty set )
# [c1, c2, c1 U c2]  or [0, c1, c2, c1 U c2] 
y_train = np.array([[0.9, 0, 0.1],
                    [0, 0.9, 0.1]])

# Test dataset with TRUE class (0 or 1)
X_test = np.array([[0.1, 0.1]])
y_test = np.array([0])

# Number of classes
nb_classes = 2
# Number of nearest neighbors
k = 2

classifier = EKNN(nb_classes, n_neighbors=k)
classifier.fit(X_train, y_train)

print(classifier.predict(X_test, return_bba=True))
precisions = classifier.score(X_test, y_test)

print("Accuracy : ", precisions)