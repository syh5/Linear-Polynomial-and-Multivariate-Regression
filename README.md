# Linear-Polynomial-and-Multivariate-Regression

# Note:

Anaconda is used for this project as it includes pandas and matplotlib.

# Linear Regression:

Linear Regression of a sample data set, "LinearRegressionData.dat", in the data folder is performed. The code initializes parameters such as step size and epochs(number of iterations), arranges the data in specified vectors and the regression is performed.

The regression function used is:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/Regress.png)

with n = 1 for this case since we are only looking at linear regression.

Cost function:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/CostFunction.png)

Gradients:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/Gradients.png)

Update Rules where α is the step size:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/UpdateRules.png)

Result:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/LinearRegression.png)

# Polynomial Regression:

Polynomial Regression of a sample data set, "PolynomialRegressionData.dat", in the data folder is performed. The code initializes parameters such as step size, regularizer and epochs(number of iterations), arranges the data in specified vectors and the regression is performed as above. The degree of the polynomial can be input after the code is run.

Regress function is same as above and the degree input decides the 'n' value.

The above Cost and Gradients are used and the following regularizers are added respectively:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/Regularizer1.png)

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/Regularizer2.png)

Update rules are the same and the result looks like:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/PolynomialRegression.png)

# Multivariate Regression and Decision Boundary:

Multivariate Regression of a sample data set, "DecisionBoundaryData.dat", in the data folder is performed. The code initializes parameters such as step size, regularizer and epochs(number of iterations), arranges the data in specified vectors and the regression is performed as above. 

Since it's a multivariate regression, a function called logistic sigmoid is used which will squash input values
to the range [0, 1], effectively modeling probability values:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/Phi.png)

and the regression function looks like:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/RegressMulti.png)

Bernoulli likelihood as cost function:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/Bernoulli.png)

The gradients turn out to be similar to that of Polynomial Regression and they can be used. The results look like:

![alt text](https://github.com/syh5/Linear-Polynomial-and-Multivariate-Regression/blob/master/images/DecisionBoundaries.png)
