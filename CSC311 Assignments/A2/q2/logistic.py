from q2.utils import sigmoid

import numpy as np


def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    weights = weights.flatten()
    y = []
    for point in data:
        point = np.append(point, 1)
        z = np.dot(weights, point)
        y.append(sigmoid(z))
    y = np.reshape(np.array(y), (-1, 1))
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          M is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    accuracy_vec = targets == np.round(y)
    ce = 0
    frac_correct = np.count_nonzero(accuracy_vec) / np.size(accuracy_vec)

    y = y.flatten()
    t = targets.flatten()
    for i in range(t.size):
        loss = -t[i]*np.log(y[i]) - (1-t[i])*np.log(1-y[i])
        ce += loss
    ce /= t.size

    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points.
           This is the same as averaged cross entropy.
           This is the objective that we want to minimize.
        df: (M + 1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    f = evaluate(targets, y)[0]
    t = targets
    data = np.insert(data, np.size(data, 1), 1, 1)
    df = []
    for j in range(weights.size):
        dloss = np.multiply(np.subtract(y, t).flatten(), data[:, j])
        dcost = np.sum(dloss) / t.size
        df.append(dcost)
    df = np.reshape(np.array(df), (-1, 1))
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          M is the number of features per example

    :param weights: A vector of weights with dimension (M + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x M, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (M+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    lamb = hyperparameters["weight_regularization"]
    weights_reg = np.delete(weights, -1)
    weights_dot = np.dot(weights_reg, weights_reg)
    reg = np.multiply(lamb/2, weights_dot)

    f = evaluate(targets, y)[0]
    t = targets
    data = np.insert(data, np.size(data, 1), 1, 1)
    df = []
    for j in range(weights.size):
        dloss = np.multiply(np.subtract(y, t).flatten(), data[:, j])
        dcost = np.sum(dloss) / t.size
        df.append(dcost+lamb*weights[j])
    df[-1] -= lamb*weights[-1]
    df = np.reshape(np.array(df), (-1, 1))
    return f+reg, df, y
