# MLAlgosANN
Simple Machine Learning algorithms based on the flow of Artificial Neural Network. The layer(s) used in the algortihms are *dense* type.

## Algorithm Flow
A typical ANN depicts the following calculations:

1. **Feed Forward** - 
The process by which an ANN produces output ![](http://www.sciweavers.org/tex2img.php?eq=%20%5Chat%7By%7D%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0). The process is governed by the component Neuron as:&nbsp;&nbsp;&nbsp;&nbsp; ![](http://www.sciweavers.org/tex2img.php?eq=z%3Dw%5Ctimes%20x%20%2B%20b&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) &nbsp;&nbsp;&nbsp;&nbsp;thus the whole process as: &nbsp;&nbsp;&nbsp;&nbsp; ![](http://www.sciweavers.org/tex2img.php?eq=%20%5Chat%7By%7D%20%3D%20f_%7Bact%7D%28%5Csum_%7Bi%3D1%7D%5En%20%20z_%7Bi%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)    where 
* ![](http://www.sciweavers.org/tex2img.php?eq=f_%7Bact%7D%28z%29%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) is the Activation function
* **w** is the weight, **b** is the bias for each neuron and
* ![](http://www.sciweavers.org/tex2img.php?eq=x&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) is the input to a neuron.

2. **Cost** - An equation ![](http://www.sciweavers.org/tex2img.php?eq=J%28%20y%2C%20%20%5Chat%7By%7D%29%20%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) to penalize the model and using the penalty to train the model. The cost function is always made such that it should be *convex* in nature, hence while performing *Gradient Descent*, the *Cost* should converge to the global minima. Typically for 
* Classification: ![](http://www.sciweavers.org/tex2img.php?eq=-%20%20%5Csum_%7Bk%3D1%7D%5Ec%20%20y_%7Bk%7D%20%5Ctimes%20log%20%28%5Chat%7By_%7Bk%7D%7D%20%29%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) where **c** is the number of classes.
* Regression: ![](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B1%7D%7B2m%7D%20%5Ctimes%20%5Csum_%7Bj%3D1%7D%5Em%20%20%28%5Chat%7By%7D_%7Bj%7D%20-%20y_%7Bj%7D%29%5E%7B2%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0) where m is the number of samples
