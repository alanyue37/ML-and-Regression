import data
from q1 import logsumexp_stable
import numpy as np
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    """
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    """
    means = np.zeros((10, 64))
    for label in range(10):
        digits = data.get_digits_by_label(train_data, train_labels, label)
        means[label] = digits.mean(0)
    return means


def compute_sigma_mles(train_data, train_labels):
    """
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    """
    covariances = np.zeros((10, 64, 64))
    means = compute_mean_mles(train_data, train_labels)

    for label in range(10):
        digits = data.get_digits_by_label(train_data, train_labels, label)
        diff = digits - means[label]
        matrix = np.matmul(diff.T, diff) / np.shape(digits)[0]
        covariances[label] = matrix + 0.01 * np.identity(64)
    return covariances


def generative_likelihood(digits, means, covariances):
    """
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    """
    logs = []
    constant = (np.shape(digits)[1]/2) * np.log(2*np.pi)

    for label in range(10):
        last = []
        middle = 0.5 * np.log(np.linalg.det(covariances[label]))
        diff = digits - means[label]
        diff_cov = np.matmul(diff, np.linalg.inv(covariances[label]))
        for i in range(np.shape(digits)[0]):
            last.append(diff_cov[i].dot(diff.T[:, i]))
        log = -constant - middle - 0.5*np.array(last)
        logs.append(log)

    return np.array(logs).T


def conditional_likelihood(digits, means, covariances):
    """
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of data points and 10 corresponds to each digit class
    """
    logs = []
    class_lik = generative_likelihood(digits, means, covariances)
    prior = np.log(1/10)
    num = class_lik + prior

    for point in num:
        log = point - logsumexp_stable(point)
        logs.append(log)
    return np.array(logs)


def avg_conditional_likelihood(digits, labels, means, covariances):
    """
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class
    label
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    s = 0
    for i in range(np.size(labels)):
        s += cond_likelihood[i][int(labels[i])]
    return s / np.size(labels)


def classify_data(digits, means, covariances):
    """
    Classify new points by taking the most likely posterior class
    """
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    return cond_likelihood.argmax(1)


def main():
    train_data, train_labels, test_data, test_labels =\
        data.load_all_data('digits')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    print(avg_conditional_likelihood(train_data, train_labels, means,
                                     covariances))
    print(avg_conditional_likelihood(test_data, test_labels, means,
                                     covariances))
    train_predictions = classify_data(train_data, means, covariances)
    test_predictions = classify_data(test_data, means, covariances)
    print(sum(train_predictions == train_labels) / np.shape(train_data)[0])
    print(sum(test_predictions == test_labels) / np.shape(test_data)[0])

    vectors = []
    for matrix in covariances:
        value = np.linalg.eig(matrix)[0].argmax()
        vector = np.linalg.eig(matrix)[1][:, value]
        vectors.append(vector.reshape(8, 8))
    frame = plt.figure()
    for i in range(len(vectors)):
        frame.add_subplot(2, 5, i+1)
        plt.imshow(vectors[i], cmap='gray')


if __name__ == '__main__':
    main()
