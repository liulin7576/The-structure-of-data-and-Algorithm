import numpy as np


def Relu(Z):  #定义一个Relu函数
    A = np.maximum(0, Z)   #max(0, Z)
    return A  

def relu_backward(dA, cache):  #计算反向传播的微分
    A1, W1, Z1, A2, W2, Z2 = cache
    
    dZ = np.multiply(dA, np.int64(A1 > 0))
    return dZ

def softmax(Z):  #softmax函数的实现
    A = np.exp(Z) / np.sum(np.exp(Z), axis = 0)
    return A

def softmax_backward(A, Y):  #计算反向传播的softmax微分
    m = Y.shape[0]
    
    A[Y.T, np.arange(m)] -= 1
    dZ2 = A.copy()
    return dZ2


    
def forward(X, Y, parameters):  #softmax前向传播
    W1 = parameters['W1']  #加载参数
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1, X.T) + b1
    A1 = Relu(Z1)  #


    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    m = Y.shape[0]
    dataLoss = -np.sum(np.log(A2[Y.T, np.arange(m)])) / m

    cache = (A1, W1, Z1, A2, W2, Z2)  #保存这些值便于backprop计算
    return dataLoss, cache




def Gradient_check(epsilon = 1e-7): #总的模型
              
    grads, X, Y, parameters = grabParameters()
    
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(grads)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    # Compute gradapprox
    for i in range(num_parameters):
        

        thetaplus = np.copy(parameters_values)                                      # Step 1
        thetaplus[i][0] = thetaplus[i, 0] + epsilon                                # Step 2
        J_plus[i], _ = forward(X, Y, vector_to_dictionary(thetaplus))                                  # Step 3

        
        thetaminus = np.copy(parameters_values)                                     # Step 1
        thetaminus[i][0] = thetaminus[i, 0] - epsilon                              # Step 2        
        J_minus[i], _ = forward(X, Y, vector_to_dictionary(thetaminus))                                  # Step 3

        

        gradapprox[i] = (J_plus[i] - J_minus[i])/(2*epsilon)


    numerator = np.linalg.norm(gradapprox - grad)                               # Step 1'
    denominator = np.linalg.norm(gradapprox) + np.linalg.norm(grad)                             # Step 2'
    difference = numerator / denominator                              # Step 3'

    if difference > 2e-7:
        print ("There is a mistake in the backward propagation! difference = " + str(difference))
    else:
        print ("Your backward propagation works perfectly fine! difference = " + str(difference))
    
    return difference


            #计算成本
    loss = compute_loss(A2, Y, parameters, regRate)
            #反向传播
    grads = backprop(X, A2, Y, cache, parameters, regRate)



    return grads


def grabParameters():  
    import pickle
    f = open(r'E:\python编程处\cs231n的作业\store_Grads.txt', 'rb')
    grads, x, y, parameters = pickle.load(f)
    return grads, x, y, parameters


def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector satisfying our specific required shape.
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2"]:
        
        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys

def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector satisfying our specific required shape.
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))

    return parameters

def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector satisfying our specific required shape.
    """
    
    count = 0
    for key in ["dW1", "db1", "dW2", "db2"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))
        
        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


Gradient_check(epsilon = 1e-7)

    








