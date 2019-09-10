import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0
   
    y[y==0] = -1
    new_X = np.insert(X, 0, 1, axis=1)
    new_w = np.insert(w, 0, b, axis=0)
    
    if loss == "perceptron":
        for i in range(max_iterations+1):
            preds = binary_predict(new_X, new_w, 999)
            preds[preds==0] = -1
            identity = (y)*preds
            identity[identity==1]= 0
            identity[identity==-1]= 1 
            m = identity * y
            d = np.dot(m,new_X) #1,D
            ad = (step_size/N)*(d)
            new_w = np.add(new_w,ad)
           
    elif loss == "logistic":
        for i in range(max_iterations+1):
            preds = y*np.dot(new_X,new_w)
            sig = sigmoid(-preds)
            d = sig*y
            m = np.dot(d,new_X)
            new_w = new_w + (step_size/N)*m
            
    else:
        raise "Loss Function is undefined."

    b = new_w[0]
    w = np.delete(new_w,0)
    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """
    value = 1 / (1 + np.exp(-z))
    return value


def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    preds = np.zeros(N)
    if b != 999:
        w = np.insert(w, 0, b, axis=0)
        X = np.insert(X, 0, 1, axis=1)
    preds = np.dot(X,w)
        
    if loss == "perceptron":
        preds[preds>0] = 1
        preds[preds<=0] = 0

    elif loss == "logistic":
        preds = sigmoid(preds)
        preds[preds>=0.5] = 1
        preds[preds<0.5] = 0

    else:
        raise "Loss Function is undefined."
    
    #assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    
    #new_X = np.insert(X, 0, 1, axis=1)
    #new_w = np.insert(w, 0, b, axis=1)
    one = np.ones((N,1))
    new_X = np.append(X, one, axis=1)
    new_w = np.append(w, np.array([b]).T, axis=1)
        
    def softmax(x):
        x = np.exp(x - np.amax(x))
        deno = np.sum(x, axis=1)
        return (x.transpose() / deno).transpose()
    
    def softmax_sgd(x):
        maxi = np.amax(x, axis=0)
        x = np.subtract(x, maxi)
        x = [np.exp(element) for element in x]
        deno = np.sum(x)
        prob = [element/deno for element in x]
        return prob
        
    
    if gd_type == "sgd":
        for i in range(max_iterations):
            r_num = np.random.choice(N)
            prob = softmax_sgd(np.matmul(new_w, new_X[r_num].T))
            prob[y[r_num]] -= 1
            g = np.matmul(np.array([prob]).T, np.array([new_X[r_num]]))
            new_w = new_w - (step_size)*g
            
    
    elif gd_type == "gd":
        y = np.eye(C)[y]
        for i in range(max_iterations):
            diff = softmax((new_w.dot(new_X.T)).T) - y
            g = np.dot(diff.transpose(), new_X) 
            new_w = new_w - (step_size/N)*g
            
    else:
        raise "Type of Gradient Descent is undefined."
    
    #b = new_w[:,0]
    #w = np.delete(new_w,0,1)
    b = new_w[:,-1]
    w = np.delete(new_w,-1,1)
    
    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    preds = np.zeros(N)
    new_w = np.insert(w, 0, b, axis=1)
    new_X = np.insert(X, 0, 1, axis=1)
    d = np.dot(new_X,new_w.transpose())
    preds= np.argmax(d, axis = 1)
    assert preds.shape == (N,)
    return preds




        