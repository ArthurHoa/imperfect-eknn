# Imperfect EK-NN

A python version of the Evidential K-Nearest Neighbors algorithm.

### Summary

When the data are labeled in an uncertain and imprecise way, the credible K-nearest neighbor model can be used for a classification problem.
The model presented here uses a computation of the gamma parameter that differs from the original model of Thierry Denœux.

### Reference

When using this code please cite and refer to the corresponding paper :  
A. Hoarau, A. Martin, J.-C. Dubois, and Y. Le Gall, “Imperfect labels with  belief  functions  for  active  learning,”  in *Belief Functions: Theory and Applications*. Springer International Publishing, 2022.

### Example

For k = 2, a new observation with its two nearest neighbors labeled *I think it's a dog* will be labeled *I think it's a dog*.

### How to use

Initialize the model by specifying the number of classes and the number *K* of neighbors :
```
classifier = EKNN(nb_classes, n_neighbors=k)
```

Train the model on the training set, with the attrributes *X_train* and the labels *Y_train* defined on $2^M$, M the number of classes :
```
classifier.fit(X_train, y_train)
```

Use score to predict the classes of *X_test*, compare them to *Y_test* and return the accuracy of the model:
```
precisions = classifier.score(X_test, y_test)
```

## Example on Credal Dog-7

### Credal Dog-7

Link to the dataset : [Credal Dog-7](https://github.com/ArthurHoa/credal-datasets)

Welsh Corgi | Collie | Shetland Sheepdog | Foxhound | Basset | Brittany | Beagle
:--:|:--:|:--:|:--:|:--:|:--:|:--:
<img src="https://github.com/ArthurHoa/credal-datasets/blob/master/ressources/pictures/Welsh_Corgi.jpg?raw=true" width="70"> | <img src="https://github.com/ArthurHoa/credal-datasets/blob/master/ressources/pictures/Collie.jpg?raw=true" width="70"> | <img src="https://github.com/ArthurHoa/credal-datasets/blob/master/ressources/pictures/Shetland_Sheepdog.jpg?raw=true" width="70"> | <img src="https://github.com/ArthurHoa/credal-datasets/blob/master/ressources/pictures/Foxhound.jpg?raw=true" width="70"> | <img src="https://github.com/ArthurHoa/credal-datasets/blob/master/ressources/pictures/Basset.jpg?raw=true" width="70"> | <img src="https://github.com/ArthurHoa/credal-datasets/blob/master/ressources/pictures/Brittany.jpg?raw=true" width="70"> |  <img src="https://github.com/ArthurHoa/credal-datasets/blob/master/ressources/pictures/Beagle.jpg?raw=true" width="70">  

### Code

```
from sklearn.model_selection import train_test_split
from eknn_imperfect import EKNN
import numpy as np

# Number of neighbors
K = 7

X = np.loadtxt('X.csv', delimiter=';')
y = np.loadtxt('y.csv', delimiter=';')
y_true = np.loadtxt('y_true.csv', delimiter=';')

indexes = [i for i in range(X.shape[0])]
train, test, _, _ = train_test_split(indexes, indexes, test_size=.2)

classes = np.array(list(set(y_true)))
nb_classes = classes.shape[0]

classifier = EKNN(nb_classes, n_neighbors=K)

classifier.fit(X[train], y[train])

precision = classifier.score(X[test], y_true[test])

print("Accuracy : ", precision)
```

Accuracy = 0.79

### Output of the model

An exmaple of the Evidential K Nearest Neighbors prediction for the following picture is given as follows:

<img src="https://www.dropbox.com/s/9fwx1gvev1h2iq2/234.jpg?raw=true" width="150">  
  
Prediction:  
m({Beagle}) = 0.67  
m({Basset}) = 0.14  
m({Berger des Shetland}) = 0.08  
m({Beagle, Basset}) = 0.01  
m({Beagle, Foxhound}) = 0.01  
m({Beagle, Epagneul Breton}) = 0.01  
m({Beagle, Foxhound, Epagneul Breton}) = 0.01  
m({Beagle, Foxhound, Berger des Shetland}) = 0.03  
m({Beagle, Basset, Foxhound, Epagneul Breton, Berger des Shetland, Corgi, Colley}) = 0.03  
  
True class: Beagle
