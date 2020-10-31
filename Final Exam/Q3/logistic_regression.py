import matplotlib.pyplot as plt
import numpy as np

A = [
    np.array([
        [0, 1, 1, 0, 0], [0, 0, 0, 1, 1], [0, 0, 0, 1, 1], [0, 1, 1, 0, 0], [1, 1, 1, 1, 0]
    ]),
    np.array([
        [1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0], [0, 1, 1, 1, 1]
    ]),
    np.array([
        [0, 0, 1, 1, 0], [0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 1, 1, 0]
    ]),
    np.array([
        [1, 1, 0, 0, 0], [0, 1, 1, 1, 1], [0, 0, 0, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 1, 1]
    ]),
    np.array([
        [1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0]
    ])
]

B = [
    np.array([
        [0, 0, 1, 1, 0], [0, 0, 1, 1, 0], [1, 0, 1, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 0, 0]
    ]),
    np.array([
        [0, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [0, 1, 1, 0, 1], [0, 0, 0, 0, 1]
    ]),
    np.array([
        [0, 1, 0, 1, 1], [0, 1, 0, 1, 1], [1, 1, 1, 0, 1], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0]
    ]),
    np.array([
        [0, 0, 0, 0, 1], [0, 1, 1, 0, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1]
    ]),
    np.array([
        [0, 1, 0, 0, 1], [1, 1, 1, 0, 1], [1, 0, 1, 0, 1], [0, 0, 1, 0, 1], [0, 0, 1, 0, 1]
    ])
]
M = [
    np.array([
        [0, 1, 1, 0, 1], [0, 1, 1, 1, 1], [1, 1, 0, 0, 0], [1, 1, 0, 1, 0], [0, 1, 0, 1, 0]
    ]),
    np.array([
        [0, 0, 1, 1, 1], [1, 0, 1, 1, 1], [0, 1, 0, 0, 1], [1, 1, 0, 1, 0], [1, 0, 0, 0, 0]
    ]),
    np.array([
        [1, 1, 0, 0, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1], [1, 1, 1, 0, 0]
    ]),
    np.array([
        [0, 1, 1, 1, 0], [0, 1, 1, 1, 0], [1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]
    ]),
    np.array([
        [1, 0, 0, 0, 0], [1, 0, 1, 1, 0], [0, 0, 0, 0, 1], [0, 1, 1, 0, 1], [0, 0, 0, 0, 0]
    ])
]

total_data = []
total_label = []
for a in A:
    total_data.append(a)
    total_label.append(0)
for b in B:
    total_data.append(b)
    total_label.append(1)

def load_dataset(train_x, train_y, M):
    train_set_x_orig = np.array(train_x)  # your train set features
    train_set_y_orig = np.array(train_y)  # your train set labels
    mystery_set_x_orig = np.array(M)

    test_x = [train_x[4], train_x[9]]
    test_y = [0, 1]
    test_set_x_orig = np.array(test_x)  # your test set features
    test_set_y_orig = np.array(test_y)  # your test set labels

    # classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    # print(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, mystery_set_x_orig


# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# initial the parameters with 0
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))  # weight
    b = 0  # bias
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


# backward/forward propagation function
def propagate(w, b, X, Y):
    """
    w -- weights, a numpy array of size (num_px * num_px * 1, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 1, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    m = X.shape[1]  # number of sample
    A = sigmoid(np.dot(w.T, X) + b)  # activation function
    cost = -np.sum(np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T)) / m  # compute cost
    dw = np.dot(X, (A - Y).T) / m  # compute gradient of weight
    db = np.sum(A - Y) / m  # gradient of bias

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw, "db": db}

    return grads, cost


# optimal function
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 1, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 1, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw  # 更新权重
        b = b - learning_rate * db  # 更新偏置

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs


# func for predict
def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 1, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 1, number of examples)
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        if A[:, i] > 0.5:
            Y_prediction[:, i] = 1
        else:
            Y_prediction[:, i] = 0

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# build and run the model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 1, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 1, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    Returns:
    d -- dictionary containing information about the model.
    """
    w, b = initialize_with_zeros(5 * 5 * 1)  # initialize
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)

    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print(Y_prediction_train)
    print(Y_prediction_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


# load and process data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, mystery_set_x_orig = load_dataset(total_data, total_label, M)
m_train = train_set_x_orig.shape[0]  # number of train sample
m_test = test_set_x_orig.shape[0]  # number of test sample
num_px = train_set_x_orig.shape[1]  # height of the picture = 5
# expand the 5 * 5 * 1 picture to a length = 25 1-D array
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
# standardize the data, used to convert the rgb or other form of the data (flatten / 255)
train_set_x = train_set_x_flatten / 1.
test_set_x = test_set_x_flatten / 1.

# train and predict based on the train and test set
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

m_mystery = mystery_set_x_orig.shape[0] # number of the mystery
mystery_set_x_flatten = mystery_set_x_orig.reshape(m_mystery, -1).T
mystery_set_x = mystery_set_x_flatten / 1.
print(predict(d["w"], d["b"], mystery_set_x))

# output a learning curve
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

