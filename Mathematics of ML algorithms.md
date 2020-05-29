# Chapter 1 

**Notes**  - Please consider installing the [Mathjax plugin](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima/) for Chrome browser if the Latex formulaes does not get rendered properly. 

(To work here and not in the ml_book repo, moved here on 28 May 2020)

1. **What will you require**
   1. Matlab or octave
   1. Install Python Anaconda from [here](https://www.anaconda.com/distribution/#download-section). 
2.  **What is Machine Learning?** 
Ability of machines to predict and learn from data. 
Arthur samuel definintion - 
Tom mitchel definition - A  computer program is said to lear from experience E with respect to some taks T and some performance measure P, if its performance on T as measured by P, improves with experience E.
3. **Based on your understanding describe what is T, P, E for a helicopter program?**
The E is leaning the experience that given a state of flying, if you put x, y, z flight parameters, can it stay afloat. P is probabiltiy that it is flying correctly. T is the task of setting the next flight parameters. 
 
4.  **How is ML different from rule based approach?** 
	1. **copied** 
	2. 
|ML| RBA |
|--|--|
| Dynamic |Static  |
| Experts optional | Experts required |
| need corpus | optional |
| training need | optional |

	3. 
5.  **Different types of ML?**
In general most ML can be classified into SL or USL. One is where we teach it and other is where it is self taught. 
**Supervised learning** Classfication based on the functions we are using .
    1. Linear regression such as *y=mx + n * 
    2. Logistic regression such as * y = discrete function of x*, such as pass or fail, while x can be the marks obtained. This is sigmoid function. This is different from linear function. In linear the transition is smooth, while in sigmoid, there is  large change within short span. 
 
    **Unsupervised learning**  - Trying to find patterns, segregate them into differnt clusters. 
    3. Auto encoding - eg PCA
    4. Clustering - 
    5. 
**Database Mining**
**Recoommender Systems**
**Reinforcement Learning**
6.  **Whate is a neural network?**
Assume functions as neurons, the central unit which takes some input and gives an output. A network of such function is called a neural network. Generalised let us say you have n inputs to f functions, giving rise to say n inputs to function f' giving output y.  

7.  **Example of neural networks?**
speech recognition, online advertising, etc. Autonomous driving. 
8. **Is neural networks same as machine learning?** 
Supervised learning in ML is same as neural networks.
9. **Types of NN**
Supervised NN, Standard NN - Real estate, advertising. 
Convulational NN - photo tagging
Recurring NN - sequence data, audio, language data
Custom NN, Hybrid NN - autonomous driving

10. **What is Structured vs unstructured data?**
schema vs no schema. SQL vs NoSql, data has very well defined meaning. USD is like audio, image, text 
11.  **How does the different types of NN deducible from each other?** 
12. **Why is deep learning taking of now**
Due to THREE things, first is NN perform better if the number of functions/neurons are high. Secondly, the amount of data in the training set is important. More the data more accurate is the performance. We did not have technologies to handle both, and now we are starting to have. Third is the algorithms being used like Sigmoid vs Relu. So DL is taking of now.
13.  **What are some blockers to solving problems **
	 1. The language used, the methodology used. For example to separate voices in the cocktail party algo, it took one line of code. However to do that in Java or other languages, it would be quite difficult. 
	 2. [W,S,V] = svd ((repmat(sum(x. *x, 1), size (x, 1), 1).*x)*x'); phew 
14. what is the language and what is the thought 



15.  **What is a cost function?**
A cost function for  $J(\theta) = \frac 1 {2m} \sum_{i=1}^m { (h_\theta (x^{(i)}) - y^{(i)})}^2$
16.  **What are the notation being used?**  
**Data set** Suppose we have a training data set, it is a 2 col table , with $x$ and $y$ values. each row forms a data set and there are $m$ rows.  We use this as the raw data and we want to predict for an x what will be the y. 
**Hypothesis** - We start with a method, a model, this part ML is not finding. It is finding the co-efficients for the model. This function is called the hypothesis.  Say this if is for a hypothesis $y = \theta_0 + x\theta_1$
**A training set** - The notation $(x^{(i)}, y^{(i)})$ means a the $i^{th}$ data set. 
**Parameters** - $\theta_1$ and $\theta_0$ are called the coefficients or parameters. 
**What is meant by** $h_\theta(x)$ -  Ofcourse this is a the hypothesis, for now it says it is a function of $x$ and $\theta$ is the paramters. *Am not sure how a function of $\alpha, \beta, \theta$ will be repesented as *. Also sometimes it is written just as $h(x)$. This is the prediction of the house price, given a $\theta_0, \theta_1 , x$
**Cost function** - Now what does $J(\theta)$ means then? 
**Our task or ML's task** - find the parameters  $\theta_0, \theta_1$ such that the cost function is minimal over the whole data set. 
**Cost function**  $J(\theta_0, \theta_1)$ This also another representation of the cost function.  It is also called squared error function. The cost function depends upon  $\theta_0, \theta_1$, a
     1. *In that light, can we also represent h as*  $h(\theta_0, \theta_1)$ because $h$ also depends upon  $\theta_0, \theta_1$ 
        1. **Answer -**  This is incorrrect, when we write $A(b)$. Say $A(b) = c_1b + c_2$ we mean that $A$ is a method dependent on $b$, we do not conside the parameters $c_1, c_2$, for a given function, the parameters are constant. It is only the $b$ that varies, not $c_1, c_2$.
        2. Thus in the same light, if a function depends upon two variables, we write $A(b, d)$, such an equation may be $A(b,d) = c_1b+ c_2d + c_3$
        3. **Cost function vs Hypothesis function difference** - For a hypothesis $h(x) = \theta_0 + \theta_1x$  and for its cost function $J(\theta) = \frac 1 {2m} \sum_{i=1}^m { (h_\theta (x^{(i)}) - y^{(i)})}^2$Note that $J$ is dependent on $\theta_0, \theta_1$ while $h$ was dependent only on $x$. So we write $J(\theta_0, \theta_1)$ while we write the hypothesis as $h(x)$. So cost function is a function of the parameters while hypothesis is function of the variable. 
        4. Note here the notation of x is $$x_j^i \equiv x_{columns/j^{th} feature}^{rows/i^{th} training set} = x_j^i= x_j^{(i)}$$  $i^{th}$ training set and $j^{th}$ feature within the $i^{th}$ training set. Training set means the rows of data we have and features mean the column. 
        6. Also noe that $h(x)$ is a linear function while $J(\theta)$ is a quadratic function. 
     2. The hypothesis $h(x) = \theta_0+\theta_1x$ is a  linear equation. It is called linear regression in one variable or univariate linear regression.
17. **When is something called regression or classification?**
When the target variable that we’re trying to predict is continuous, such as in our housing example, we call the learning problem a regression problem. When y can take on only a small number of discrete values (such as if, given the living area, we wanted to predict if a dwelling is a house or an apartment, say), we call it a classification problem. **copied from coursera**
19. Question from coursera
Suppose we have a training set with m=3 examples, plotted below. Our hypothesis representation is  $h_\theta(x) = \theta_1x$, with parameter  $\theta_1$. The cost function  $J(\theta_1)J(θ1​)$  is  $J(\theta_1) = \frac{1}{2m} \sum^m_{i=1} (h_\theta (x^{(i)}) - y^{(i)})^2$. What is  $J(0)$?

![A plot of $$h_\theta(x)$$ versus $$x$$. There are three markers on the plot: one at (1,1), one at (2,2), and one at (3,3).](http://spark-public.s3.amazonaws.com/ml/images/2.3-quiz-1-fig.jpg)
0, 1/6, 1, 14/6
The answer to this is if $J(0)$, Then we are considering $\theta$ as 0. So y for any x  = 0 is our hypothesis. In that case, the cost function = $1/2m$ of $n^2$ i.e. 14/6

19.  **Algorithmic way to find the gradient descent - Batch gd**
Batch gradient descent - Here we use iterations and get to J.
repeat untill convergence { $$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)$$} and simultaneously update $\theta_0, \theta_1$.
$\alpha$ is the learning rate.
  

20.  **Generic notion of a linear hypothesis**
A generic $h$ is as below. 
$h = \theta_0 + \theta_1x_1 + \theta_2x_2 + \theta_3x_3 + ...$
In short this would be 
$$h = \sum_{i=0}^nx_i\theta_i$ 


So the vector x is $\begin{bmatrix}x_0\\\ x_1\\\ x_2\\\ x_3\\\ x_n\end{bmatrix}$ and the $\theta$ vector is $\begin{bmatrix}\theta_0 \\\ \theta_1 \\\  \theta_2 \\\ ...  \\\ \theta_n\end{bmatrix}$
The dimension of x is $n+1$ and the dimension of $\theta$ is $n+1$ where $n$ is the number of training set. 

So the hypothesis can be represented as $h = \sum_{i=0}^n\theta^Tx$ (note - not the other way round in which case it will give a nxn matrix)


## Week 2

21.  **Mathematics of gradient decent with multiple variables.** 
The formulae is 
**repeat until convergence** {
$\theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}).x_0^{(i)}$ and 
$\theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}).x_1^{(i)}$
}

this is because 
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

where  
$$\frac{\partial}{\partial \theta_j} J(\theta) = \frac{1}{m}\sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}).x_0^{(i)}$$

23. Ways to speed up gradient descent
	   1. **feature scaling** Make all the features (variables) of the same range. This is $\frac{x }{x_{max}-x_{min}}$. 
	   2.  **mean normalisation** - use the formula $\frac{(x - \bar x)}{x_{max}-x_{min}}$ . so FS = MN - $\frac{ \bar x}{x_{max}-x_{min}}$ 
	       1.  Using features scaling - Using this all the variables range between $-1\leq x\leq 1$, the ranges may be different for different variables however will be in order (not much bigger or much smaller), and descent is faster. 
	   4.  **Choice of $\alpha$** - The reason is $\theta$ will descend quickly on small ranges and slowly on large ranges and will oscillate if variables are uneven. 
	4. **How to choose features?** = create secondary features from primary ones if required.  so for example, instead of taking length and breadth , you take area. 
		1. There is an area called **model selection algorithm** that is essentially tells you which features to keep and not keep. 
		2. Note throwing away features = through away data.  Another way is to **regularization**. WHich is keep the theta low, and do not throw it away. 
		3. Go through all the features and take make assumptions. 
	6. **bring higher order polynomial to lower one** - instead of using $x^2$ or $x^3$, if we have to use both, we use $x_1 = x^2$ ; $x_2 = x^3$; $x_3 = x^{1/3}$  so our features themselves becomes linear.
	7. **Too many variables** - delete them or ignore if possible or comibine. Remove redundant variables or those already considered.  
	8. 
24. If $J(\theta)$ is a polynomial function of $\theta$ then to get $\theta$ for which it is minimum is to solve the equation $\frac{dJ}{d\theta}$. However, if the function is  $J (\theta_0, \theta_1, \theta_2, ...)$ or simply put $J(\theta)$  that is it is multivariable function, then to find the minimum requires partial derivatives as below $$\frac {\partial }{\partial \theta_j} J(\theta) = 0 $$  and we have solve it. We get $j$ equations to solve and we solve it. 
25. How to solve for multivariable function in matrix way. 
$$\theta = (X^TX)^{-1}X^Ty$$ where X = 
$\begin{bmatrix}1 &x_1 \\\ 1 &x_2 \\\ ...&... \\\ 1 &x_{n-1} \\\ 1 &x_n\end{bmatrix}$ and y is $\begin{bmatrix}y_1\\\ y_2\\\ ...\\\ y_{n-1}\\\ y_n\end{bmatrix}$ 

**In octave this command is** - $pinv(X' *X) *X' * Y$

26.  Linear regression used for classification problem
     1. $P(y = 0 |x; \theta)$ means the probablity that y = 0, for a feature x and parameter $\theta$
     2.  This does not scale well. Hence we use the sigmoid functin. 
27.  Decision boundaries - these are boundaries on either side of which the hypothesis has positive or 1 or a certain set of classification and on the other side it is -negative or -1 or the other side of the classification. 
28. **Sigmoid function** = **Logistic Function** = is given by 
    1. $h_\theta (x) = g ( \theta^T x )$ and
    2. $z = \theta^T x$ and
    3.  $g(z) = \dfrac{1}{1 + e^{-z}}$ 
    4. How this works  - if x is +ve then then Y = 1 and if it is -ve then Y = 0
    5. the limits 
		  1. $z=0, e^{0}=1 \Rightarrow g(z)=1/2$
		  2. $z \to \infty, e^{-\infty} \to 0 \Rightarrow g(z)=1$
		  3. $z \to -\infty, e^{\infty}\to \infty \Rightarrow g(z)=0$
    6. The way to find $\theta$ is 
    $$\theta :=\theta - \frac\alpha {m} X^T(g(X\theta) - \overrightarrow  y) $$
    7. Logistic regression cost function - 
    $j(\theta) = - 1/m[\sum_{i=1}^my^{(i)}logh_\theta(x^{(i)}) + (1 - y^{(i)}) log(1 - h_\theta(x^{(i)}))]$
29. Calculating $min J(\theta)$ for logistic regress. Octave function **fminunc ** does this. 
	1. it takes in   options arg.  **option = ("GradObj", "on", "MaxIter", 100)**
	2. [optTheta, functionVal, exitFlag] = fminuc(@costFunction, intiialTheta, options). 
	3. Where we specify some initial theta and a cost function. 
30. **One vs all** - variation of Logistic regression. Multi Class classification -  Here we train a  logistic regressions one for each cluster of data. For the cluster, we train the LogR = true if data belogns to cluster and false if not. Then we go ahead and train other LogR for other clusters. Finally we run the prediction set, the LogR which predicted high is the answer.
![enter image description here](https://i.imgur.com/RqmnOg8.png)  
	 1. The gradient descent of LogR
![Various interpretations of GD for LogR](https://i.imgur.com/Z7K2vYT.png)
    
31. Algos for LogR - 
     1. Gradient descent. This one we discussed, rest are sometimes faster, advanced, and more complex. They do not require a $\alpha$ to be chosen.
     9.  Conjugate descent
    10. BFGS
    11. LFGS 
32.  **Overfitting** - Whent the model is counter intuitive, contours itself too much to fit the data exactly, and rather fails to give a proper prediction , we have over fitting problem. 
33.  What is Bias - the model predicts something else  most of the time while training data is otherwise.
34. **regulariztion parameter** An extra function added to the cost funtion to take care of theta wihch are insignificant. However, if we make this large, then it overshadows other parameters. 
35. **Algo** - **Regularized linear regression** 
Cost function for LogR 
$\theta_0 := \theta_0 - \alpha\ \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_0^{(i)}$
$\theta_j := \theta_j - \alpha\ \left[ \left( \frac{1}{m}\ \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} \right) + \frac{\lambda}{m}\theta_j \right]$

36. NN
	1. Terminology 
	2. activation, hidden layer , output, input, weights
	3. weights are matrix , it is like a graph of vertices and edges, the weights are edges weight
	4. the dimesion of $\theta$ - Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of Θ(1) is going to be 4×3 where sj=2s_j = 2sj​=2 and sj+1=4s_{j+1} = 4sj+1​=4, so sj+1×(sj+1)=4×3s_{j+1} \times (s_j + 1) = 4 \times 3sj+1​×(sj​+1)=4×3.
38. 

# Week 5 - 
1. We are trying to do a multi category classification problem using a NN.  
2. We know the Logistics regression cost function as 
    $j(\theta) = - 1/m[\sum_{i=1}^my^{(i)}logh_\theta(x^{(i)}) + (1 - y^{(i)}) log(1 - h_\theta(x^{(i)}))] + regularization term$
    The regularizatoin term is as below 
    $+\frac{\lambda}{2m}\theta_j\sum_{j=1}^n\theta^2_j$
3.  So for a neural network  defined by 
$h_\Theta(x)\subset R^K$ where $h_\Theta(x)=i^{th}$ and output the cost function  becomes
$J(\Theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2$

Here 
-   L = total number of layers in the network
-   $s_l$​  = number of units (not counting bias unit) in layer l
-   K = number of output units/classes
4. Again our target is to optimize the cost function, implement it and find the values for $\theta$
5.  Back propogation algorithm - this is the algorithm to find the optimized $J(\Theta)$. We need to find $$



## Week 6 - Diagnostics and Debugging
### Debugging Machine learning algos
1. Collect more training data - What is an apt quantity of training data. 
2. Try small set of features
3. Get more features. 
4. Add polynomial features
5. Decreasing or increase $\lambda$

![enter image description here](https://i.imgur.com/3STBeFl.png)

All these may work or not work. So how do we know where the prob is?

**Diagnostic ** - takes time, however directs you to actual issue. 
### Over fitting
How do we know if it is overfitting - plot if the feature set is small. 
**Way 1** If not split the data into train and test set. A typical split is 70:30. Then compute the error for traiing and test set. If traiing is << 0 and test is >>0 you have over fitting. 
**Way 2 ** Split into 3 sets , training, cross validation, test set. 60:20:20. So then the errors are 
![enter image description here](https://i.imgur.com/ndeL9Jo.png)

Fitting higher degree polynomial and its effects on train error and cross validation error 
![enter image description here](https://i.imgur.com/4RVwcpc.png)

How to distinguish high bias vs high variance. 
If both J train and J cv both are high then we have bias, if J train is low and J cv is high we have variance issue
![enter image description here](https://i.imgur.com/9eIwwGO.png)

The regions of under and over fitting is show in this graph more clearly.
![enter image description here](https://i.imgur.com/yPNi4s7.png)

**Lambda and bias -** 
high $\lambda$ means large bias. 
small $\lambda$ less bais, effectively means lambda effect is low. 

![enter image description here](https://i.imgur.com/0brkDAO.png)

**Learning curves** - These are also used to diagnose ML problems. 
This is the size of error with respect to growing number of items in the training set. 
The cv error - also behaves inversely. The more data we have the better the fit is  
![enter image description here](https://i.imgur.com/DbP1qv3.png)

So this is the inference we are trying to draw, for high bias, more data is not going to help.
![enter image description here](https://i.imgur.com/QqDMZxZ.png)


For variance, it does help. 
![enter image description here](https://i.imgur.com/dnjoeMV.png)

So summary 
![enter image description here](https://i.imgur.com/yKwm2RM.png)

More layers are computationally expensive and prone to overfitting
![enter image description here](https://i.imgur.com/lkUnDop.png)

# Deciding What to Do Next Revisited

Our decision process can be broken down as follows:

-   **Getting more training examples:**  Fixes high variance
-   **Trying smaller sets of features:**  Fixes high variance
-   **Adding features:**  Fixes high bias
-   **Adding polynomial features:**  Fixes high bias
-   **Decreasing λ:**  Fixes high bias
-   **Increasing λ:**  Fixes high variance.

### **Diagnosing Neural Networks**

-   A neural network with fewer parameters is  **prone to underfitting**. It is also  **computationally cheaper**.
-   A large neural network with more parameters is  **prone to overfitting**. It is also  **computationally expensive**. In this case you can use regularization (increase λ) to address the overfitting.

Using a single hidden layer is a good starting default. You can train your neural network on a number of hidden layers using your cross validation set. You can then select the one that performs best.

**Model Complexity Effects:**

-   Lower-order polynomials (low model complexity) have high bias and low variance. In this case, the model fits poorly consistently.
-   Higher-order polynomials (high model complexity) fit the training data extremely well and the test data extremely poorly. These have low bias on the training data, but very high variance.
-   In reality, we would want to choose a model somewhere in between, that can generalize well but also fits the data reasonably well.

### Prioritizing Work flow

<!--stackedit_data:
eyJoaXN0b3J5IjpbODg5MTM4Mjc4LC0zMDM2MjQ1MTUsMTQxMz
cyMTQ0OSwxMDYxMTg5OTcxLDgzMjYxNjgzOSwxMDkxMTkyNDA4
LC0xMjEzODEyMzE0LDI5Nzc4NTUzNiwzMjAyOTA2NzksMTMyMz
I5ODE3OCwxMTg1OTY1NzE2LC0yNjE1Njg0MTIsLTI2MTU2ODQx
MiwxMzcxMTQwMTc1LDEzNzExNDAxNzUsMjAxNTM1MTYzMywyMD
E1MzUxNjMzLC05NDI4MTg0MV19
-->