import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import math

# meta-parameters for program
trial_name = 'p1_fit' # will add a unique sub-string to output of this program
degree_input = input('Enter the degree:') 
degree = int(degree_input) # p, order of model
beta = 0.01 # regularization coefficient
alpha = 0.01 # step size coefficient
eps = 0.00001 # controls convergence criterion
n_epoch = 10000 # number of epochs (full passes through the dataset)

# begin simulation

def regress(X, theta):
 
    f = theta[0]+np.dot(X,np.transpose(theta[1]))   #Regression function
    return f

def gaussian_log_likelihood(mu, y):
	G = pow((y-mu),2)  #Gaussian log likelihood
	return G
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood
    
    f = regress(X,theta)
    G = gaussian_log_likelihood(f,y)
    cost = np.sum(G)
    thetaSquares = np.sum(np.square(theta[1]))
    
    return (0.5/len(X))*(cost + beta * thetaSquares)
    

	
def computeGrad(X, y, theta, beta):
    
    regress_y = regress(X,theta)-y
    don = np.multiply(regress_y,X)
    dL_dw = np.sum(don,axis = 0)
    dL_db = np.sum(regress_y) 
    
    nabla = ((dL_db)/len(X), (dL_dw+beta*theta[1])/len(X)) # nabla represents the full gradient
    return nabla

path = os.getcwd() + '/data/PolynomialRegressionData.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

X_input = np.zeros([len(X),degree])
for i in range(len(X)):
    for j in range(degree):
        X_input[i,j] = np.power(X[i],j+1)
        

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros([1,degree])
b = np.array([0])
theta = (b,w)

L = computeCost(X_input, y, theta, beta)
previous_L = 0
print("-1 L = {0}".format(L))
i = 0
while(i < n_epoch):
    
    dL_db, dL_dw = computeGrad(X_input, y, theta, beta)
    b = theta[0]
    w = theta[1]
    	# update rules
        
    b = b - alpha*dL_db
    w = w - alpha*dL_dw

    L = computeCost(X_input, y, theta, beta)
    if math.fabs(previous_L - L) < eps:
        break;
    
    theta = (b,w)
    	
    	
    print(" {0} L = {1}".format(i,L))
    i += 1
    previous_L = L
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test, axis=1) # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))
X_mat = np.zeros([len(X_feat),degree])
for i in range(len(X_feat)):
    for j in range(degree):
        X_mat[i,j] = np.power(X_feat[i],j+1)
# apply feature map to input features x1

plt.plot(X_test, regress(X_mat, theta), label="Model")
plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")

plt.show()
