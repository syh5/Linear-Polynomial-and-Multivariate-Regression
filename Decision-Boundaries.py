import os  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

# meta-parameters for program
trial_name = 'p6_reg0' # will add a unique sub-string to output of this program
degree = 6 # p, degree of model 
beta = 1 # regularization coefficient
alpha = 0.01 # step size coefficient
n_epoch = 10000 # number of epochs (full passes through the dataset)
eps = 0.0 # controls convergence criterion

# begin simulation

def sigmoid(z):     # Used to estimate Bernoulli parameter
    phi = 1/(1 + np.exp(-z))
    return phi

def predict(X, theta):      # Discretize the probability to either 1 or 0 based on a threshold
    prediction = np.zeros([X.shape[0],1])
    predict = regress(X,theta)
    for i in range(len(predict)) :
        if predict[i] > 0.5:
            prediction[i] = 1
        else:
            prediction[i] = 0
    return prediction
	
def regress(X, theta):
    f = sigmoid(theta[0]+np.dot(X,np.transpose(theta[1])))
    return f

def bernoulli_log_likelihood(p, y):
    log_1 = np.log(p)
    log_2 = np.log(1-p)
    G1 = np.multiply(-y2,log_1)
    G2 = np.multiply(-(np.ones([y2.shape[0],y2.shape[1]])-y2),log_2)
    
    return G1+G2
	
def computeCost(X, y, theta, beta): # loss is now Bernoulli cross-entropy/log likelihood

    f = regress(X,theta)
    G = bernoulli_log_likelihood(f,y)
    cost = np.sum(G)
    thetaSquares = np.sum(np.square(theta[1]))
    
    return (0.5/len(X))*(cost + beta * thetaSquares)
	
def computeGrad(X, y, theta, beta): 
    
    regress_y = regress(X,theta)-y2
    don = np.multiply(regress_y,X)
    dL_dw = np.sum(don,axis = 0)
    dL_db = np.sum(regress_y) 
    nabla = ((dL_db)/len(X), (dL_dw+beta*theta[1])/len(X)) # nabla represents the full gradient
    return nabla
	
path = os.getcwd() + '/data/DecisionBoundaryData.dat'  
data2 = pd.read_csv(path, header=None, names=['Test 1', 'Test 2', 'Accepted'])

positive = data2[data2['Accepted'].isin([1])]  
negative = data2[data2['Accepted'].isin([0])]
 
x1 = data2['Test 1']  
x2 = data2['Test 2']

# apply feature map to input features x1 and x2
cnt = 0
for i in range(1, degree+1):  
	for j in range(0, i+1):
		data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
		cnt += 1

data2.drop('Test 1', axis=1, inplace=True)  
data2.drop('Test 2', axis=1, inplace=True)

# set X and y
cols = data2.shape[1]  
X2 = data2.iloc[:,1:cols]  
y2 = data2.iloc[:,0:1]

# convert to numpy arrays and initalize the parameter array theta
X2 = np.array(X2.values)  
y2 = np.array(y2.values)  
w = np.zeros((1,X2.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X2, y2, theta, beta)
print("-1 L = {0}".format(L))
i = 0
halt = 0

while(i < n_epoch and halt == 0):
    dL_db, dL_dw = computeGrad(X2, y2, theta, beta)
    b = theta[0]
    w = theta[1]
    	# update rules
        
    b = b - alpha*dL_db
    w = w - alpha*dL_dw
    
    L = computeCost(X2, y2, theta, beta)
    theta = (b,w)
    	
    print(" {0} L = {1}".format(i,L))
    i += 1
# print parameter values found after the search
print("w = ",w)
print("b = ",b)

predictions = predict(X2, theta)
compare = predictions == y2
compare = compare*1
right_count = 0

for i in range(len(compare)):
    if compare[i] == 1:
        right_count += 1

wrong_count = 118 - right_count
accuracy = (right_count/118)*100
# compute error (100 - accuracy)
err = 100 - accuracy
#print 'Error = {0}%'.format(err * 100.)
print('Error = {0}%'.format(err))

# make contour plot
xx, yy = np.mgrid[-1.2:1.2:.01, -1.2:1.2:.01]
xx1 = xx.ravel()
yy1 = yy.ravel()
grid = np.c_[xx1, yy1]
grid_nl = []
# re-apply feature map to inputs x1 & x2
for i in range(1, degree+1):  
	for j in range(0, i+1):
		feat = np.power(xx1, i-j) * np.power(yy1, j)
		if (len(grid_nl) > 0):
			grid_nl = np.c_[grid_nl, feat]
		else:
			grid_nl = feat
probs = regress(grid_nl, theta).reshape(xx.shape)

f, ax = plt.subplots(figsize=(8, 6))
ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

ax.scatter(x1, x2, c=y2, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-1.5, 1.5), ylim=(-1.5, 1.5),
       xlabel="$X_1$", ylabel="$X_2$")

plt.show()
