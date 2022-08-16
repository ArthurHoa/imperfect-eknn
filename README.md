# Imperfect EK-NN

A python version of the Evidential K-Nearest Neighbors algorithm.

### Summary

When the data are labeled in an uncertain and imprecise way, the credible K-nearest neighbor model can be used for a classification problem.
The model presented here uses a computation of the gamma parameter that differs from the original model of Thierry Denœux.

### Reference

When using this code please cite and refer to [Paper being published](https://github.com/ArthurHoa/imperfect_eknn)

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
