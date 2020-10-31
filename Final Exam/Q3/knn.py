import matplotlib.pyplot as plt
import numpy as np

"""
black is 1, white is 0
A contains 5 A pic
B contains 5 B pic
M contains 5 mystery pic
each picture denotes by 5 * 5 numpy array
"""
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
for i in range(5):
#for a in A:
    total_data.append(A[i])
    total_label.append(0)
#for b in B:
    total_data.append(B[i])
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


# Euclidean Distance
def distance(X_test, X_train):
    """
    Arguments:
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 1, m_test)
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 1, m_train)
    Returns:
    distances -- distance between data in test set and data in train set by a numpy of shape (num_test_data, num_train_data)
    """
    num_test = X_test.shape[1]
    num_train = X_train.shape[1]
    distances = np.zeros((num_test, num_train))
    # (X_test - X_train)*(X_test - X_train) = -2X_test*X_train + X_test*X_test + X_train*X_train
    dist1 = np.multiply(np.dot(X_test.T, X_train), -2)  # -2X_test*X_train, shape (num_test, num_train)
    dist2 = np.sum(np.square(X_test.T), axis=1, keepdims=True)  # X_test*X_test, shape (num_test, 1)
    dist3 = np.sum(np.square(X_train), axis=0, keepdims=True)  # X_train*X_train, shape(1, num_train)
    distances = np.sqrt(dist1 + dist2 + dist3)

    return distances


def predict(X_test, X_train, Y_train, k):
    """
    Predict whether the label is 0 or 1 using learned knn
     Arguments:
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 1, m_test)
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 1, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    k -- number of neighbors of each train data sample
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    distances -- a numpy array (vector) containing distance between data in test set and data in train set
                in shape of(num_test_data, num_train_data)(num_test_data, num_train_data)
    """
    distances = distance(X_test, X_train)
    num_test = X_test.shape[1]
    Y_prediction = np.zeros(num_test)
    for i in range(num_test):
        # sort in ascending order by e_distance and choose the minumum-kth distance points
        dists_min_k = np.argsort(distances[i])[:k]
        y_labels_k = Y_train[0, dists_min_k]  # determine the class which the first kth points are in
        # return the highest rate class of the first kth points as the prediction
        Y_prediction[i] = np.argmax(np.bincount(y_labels_k))

    return Y_prediction, distances


# build and run the model
def model(X_test, Y_test, X_train, Y_train, k, print_correct=False):
    """
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 1, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 1, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    k -- number of neighbors of each train data sample
    print_correct -- if true, stdout accuracy of model
    Returns:
    d -- dictionary containing information about the model.
    """
    Y_prediction, distances = predict(X_test, X_train, Y_train, k)
    num_correct = np.sum(Y_prediction == Y_test)
    accuracy = np.mean(Y_prediction == Y_test)
    if print_correct:
        print('Correct %d/%d: The test accuracy: %f' % (num_correct, X_test.shape[1], accuracy))
    d = {"k": k,
         "Y_prediction": Y_prediction,
         "distances": distances,
         "accuracy": accuracy}
    return d


# load and process data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, mystery_set_x_orig = load_dataset(total_data, total_label,
                                                                                              M)
m_train = train_set_x_orig.shape[0]  # number of train sample
m_test = test_set_x_orig.shape[0]  # number of test sample
num_px = train_set_x_orig.shape[1]  # height of the picture = 5
# expand the 5 * 5 * 1 picture to a length = 25 1-D array
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
# standardize the data, used to convert the rgb or other form of the data (flatten / 255)
train_set_x = train_set_x_flatten / 1.
test_set_x = test_set_x_flatten / 1.

m_mystery = mystery_set_x_orig.shape[0]  # number of the mystery
mystery_set_x_flatten = mystery_set_x_orig.reshape(m_mystery, -1).T
mystery_set_x = mystery_set_x_flatten / 1.

for k in range(1, 6):
    print("k =", k)
    # train and predict based on the train and test set
    d = model(test_set_x, test_set_y, train_set_x, train_set_y, k, print_correct=True)

    Y_predict, Y_distance = predict(mystery_set_x, train_set_x, train_set_y, k)
    print(Y_predict)


