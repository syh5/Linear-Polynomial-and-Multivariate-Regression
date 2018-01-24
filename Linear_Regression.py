import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

# NOTE: you will need to tinker with the meta-parameters below yourself (do not think of them as defaults by any means)
# meta-parameters for program
alpha = 0.01 # step size coefficient
eps = 0.0 # controls convergence criterion
n_epoch = 70 # number of epochs (full passes through the dataset)

# begin simulation

def regress(X, theta):
    
    f = theta[0]+np.dot(X,np.transpose(theta[1]))   #Regression function
    return f

def gaussian_log_likelihood(mu, y):
	G = pow((y-mu),2)  #Gaussian log likelihood
	return G
	
def computeCost(X, y, theta): # loss is now Bernoulli cross-entropy/log likelihood
  
    f = regress(X,theta)
    G = gaussian_log_likelihood(f,y)
    cost = np.sum(G)
        
    return (0.5/len(X))*cost
	
def computeGrad(X, y, theta): 
   
    regress_y = regress(X,theta)-y
    don = np.multiply(regress_y,X)
    dL_dw = np.sum(don,axis = 0)
    dL_db = np.sum(regress_y) 
    
    nabla = ((dL_db)/len(X), (dL_dw)/len(X)) # nabla represents the full gradient
    return nabla

path = os.getcwd() + '/data/LinearRegressionData.dat'  
data = pd.read_csv(path, header=None, names=['X', 'Y']) 


# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)

# convert to numpy arrays and initalize the parameter array theta 
w = np.zeros((1,X.shape[1]))
b = np.array([0])
theta = (b, w)

record_L = np.zeros([n_epoch,1])
L = computeCost(X, y, theta)
print("-1 L = {0}".format(L))
L_best = L
i = 0
cost = [] 
while(i < n_epoch):
    dL_db, dL_dw = computeGrad(X, y, theta)
    b = theta[0]
    w = theta[1]
	# update rules

    b = b - alpha*dL_db
    w = w - alpha*dL_dw

    L = computeCost(X, y, theta) # track our loss after performing a single step
    record_L[i] = L
    
    theta = (b,w)
    	
    print(" {0} L = {1}".format(i,L))
    i += 1
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

kludge = 0.25 # helps with printing the plots
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_test = np.expand_dims(X_test, axis=1)

plt.figure(1)
plt.plot(X_test, regress(X_test, theta), label="Model")
plt.scatter(X[:,0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")

# visualize the loss as a function of passes through the dataset

plt.savefig('bleh.png')
plt.show() # convenience command to force plots to pop up on desktop

plt.figure(2)
plt.plot(np.arange(0,n_epoch),record_L, label="Cost with number of epochs")
plt.xlabel("Number of epochs")
plt.ylabel("Cost")
plt.legend(loc="best")
plt.savefig('meh.png')




