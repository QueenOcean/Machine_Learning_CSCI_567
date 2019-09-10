"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    err = None
    if np.size(X,1) == np.size(w,0):
        d = np.dot(X,w)
    else:
        d = np.dot(X.transpose(),w)
    s = np.subtract(d, y)
    sq = np.square(s)
    err = np.mean(sq, dtype=np.float64)
    """
    N = len(X)
    s = 0
    l =[]
    for i in range(N):
        n=w.size
        d=X[i][0].size
        if n == d:
            s = np.square(np.dot(X[i], w) - y[i])
        else:
            s = np.square(np.dot(X[i].transpose(), w) - y[i])
        l.append(s)
        #s = s+ np.square(np.dot(X[i], w) - y[i])
        #print s
    #err = np.float64(s/N)
    err = np.mean(l, dtype=np.float64)
    """
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################		
  w = None
  tr = X.transpose()
  d = np.dot(tr,X)
  inv = np.linalg.inv(d)
  xy = np.dot(tr, y)
  w = np.dot(inv, xy)
  """
  result = []
  for i in range(D):
        b = tr[i:i+1,:]
        bt = b.tranpose()
        d = np.dot(b,bt)
        mat = np.matmul(b,y)
        res = np.dot(d,mat)
        result.append(res)
  w=result   
  """
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    w = None
    tr = X.transpose()
    M = np.dot(tr,X)
    eigenValues, eigenVectors = np.linalg.eig(M)
    mn = np.amin(np.absolute(eigenValues))
    N = np.size(X,1)
    I = 0.1*np.identity(N)
    while mn < 0.00001:
        M = np.add(M,I)
        eigenValues, eigenVectors = np.linalg.eig(M)
        mn = np.amin(np.absolute(eigenValues))
    xy = np.dot(tr,y)
    inv = np.linalg.inv(M)
    w = np.dot(inv, xy)
    return w


###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    w = None
    tr = X.transpose()
    M = np.dot(tr,X)
    N = np.size(X,1)
    I = lambd*np.identity(N)
    M = np.add(M, I)
    xy = np.dot(tr,y)
    inv = np.linalg.inv(M)
    w = np.dot(inv, xy)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################		
    bestlambda = None
    min_error = 9999
    lambd = 10**-19
    power = -19
    while lambd < 10**19:
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        err = mean_square_error(w, Xval, yval)
        if err < min_error:
            min_error = err
            bestlambda = lambd
        power = power +1
        lambd = 10**(power)
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    T=X
    if power == 1:
        return X
    for i in range(2, power+1):
        M = np.power(T, i)
        X= np.concatenate((X,M), axis=1)
    return X


