# MLAlgosANN
Simple Machine Learning algorithms based on the flow of Artificial Neural Network. The layer(s) used in the algortihms are *dense* type.

## Algorithm Flow
A typical ANN depicts the following calculations:

1. **Forward Propagation** - 
The process by which an ANN produces output **y^** and is governed by each component neuron **w.x + b**. Hence, **y^ = f<sub>act</sub>(Summation(w.x + b))** where
* **f<sub>act</sub>** is the activation function. It is one of the factors that changes Algorithm.
* **w, b** are the weight and bias of a neuron.
* **x** is the input fed to a neuron.

2. **Cost** - An equation **J(y^,y)** to penalize the model and using the penalty to train the model. The cost function is always made such that it should be *convex* in nature, hence while performing *Gradient Descent*, the *Cost* should converge to the global minima. Typically for 
* Classification - **J(y^,y) = Summation(y.log(y^))** for all *c* classes.
* Regression - **J(y^,y) = (1/2m).Summation((y^-y)<sup>2</sup>)** for all *m* samples.

3. **Backward Progagation** - The process of learning and updating the learnable parameters in order to achieve maximum possible accuracy and minimum possible cost(or loss). The whole process is governed by *Gradient Descent* that minimizes the Cost function by taking specified number of steps to reach a global minima. *Gradient Descent* is evaluated by chain rule as :
* Weights : **dW** or **dJ/dw = (dJ/dy^) x (dy^/dz) x (dz/dw)**
* Bias : **dB** or **dJ/dw = (dJ/dy^) x (dy^/dz) x (dz/db)**, since **J=f(y^)**, **y^=f<sub>act</sub>(z)** and **z=f(w,b)** 

4. **Parameters Update** - As learnable parameters keep updating to achieve a desired accuracy, the model keeps learning. Parameters are updated as :
* Weights : **W<sub>new</sub> = W<sub>old</sub> - learningRate x dW** where *learningRate* is another hyper parameter that acts as a step(large or small) scaled of the descent to reach the global minima. When kept too large, it bounces back and forth off the global minima, hence always passing forth the global minima. However, when kept too small, it enables many descents to reach the global minima thereby slowing down the learning. Hence the parameter should be adjusted accordingly.
* Bias : Similarly **B<sub>new</sub> = B<sub>old</sub> - learningRate x dB**

## Project Setup
The project is made in Scala using Nd4J, a scientific computing library for the JVM with routines designed to run fast with minimum RAM requirements. It is a typical sbt project hence:
1. Install and setup sbt.
2. Download or clone the repository.
3. Open terminal at the downloaded(or cloned) directory.
4. Command: sbt run and specify the class you want to run.

**NOTE** - 
* In order to tune an algorithm for a dataset, play around with hyperparameters specified for each algorithm.
* It may be required to implement initialization and data transformation techniques for better results, but the sole purpose of the project is to depict the way algorithms work in an ANN. 
