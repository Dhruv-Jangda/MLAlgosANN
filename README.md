# MLAlgosANN
Simple Machine Learning algorithms based on the flow of Artificial Neural Network. The layer(s) used in the algortihms are *dense* type.

## Algorithm Flow
A typical ANN depicts the following calculations:

1. **Feed Forward** - 
The process by which an ANN produces output **y^** and is governed by each component neuron **w.x = b**. Hence, **y^ = f<sub>act</sub>(Summation(w.x = b))** where
* **f<sub>act</sub>** is the activation function. It is one of the factors that changes Algorithm.
* **w, b** are the weight and bias of a neuron.
* **x** is the input fed to a neuron.

2. **Cost** - An equation **J(y^,y)** to penalize the model and using the penalty to train the model. The cost function is always made such that it should be *convex* in nature, hence while performing *Gradient Descent*, the *Cost* should converge to the global minima. Typically for 
* Classification - **J(y^,y) = Summation(y.log(y^))** for all *c* classes.
* Regression - **J(y^,y) = (1/2m).Summation((y^-y)<sup>2</sup>)** for all *m* samples.
