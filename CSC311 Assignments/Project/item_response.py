from utils import *

import numpy as np
import matplotlib.pyplot as plt

from utils import _load_csv


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    for i in range(len(data['is_correct'])):
        theta_i = theta[data['user_id'][i]]
        beta_j = beta[data['question_id'][i]]
        diff = theta_i - beta_j
        if data['is_correct'][i] == 1:
            ll_ij = diff - np.log(1+np.exp(diff))
        else:
            ll_ij = np.log(1 - np.exp(diff)/(1+np.exp(diff)))
        log_lklihood += ll_ij
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood / len(data['is_correct'])


def update_theta_beta(sparse_matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param sparse_matrix: A sparse matrix representation of the data
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    matrix = sparse_matrix.toarray()
    theta_grad = []
    for i in range(len(theta)):
        ones = np.where(matrix[i] == 1)
        beta_ones = np.exp(beta[ones])
        der_ones = np.sum(beta_ones / (np.exp(theta[i]) + beta_ones))
        zeros = np.where(matrix[i] == 0)
        beta_zeros = np.exp(beta[zeros])
        der_zeros = np.sum(-np.exp(theta[i]) / (np.exp(theta[i]) + beta_zeros))
        theta_grad.append(der_ones + der_zeros)
    theta += lr*np.array(theta_grad)
    beta_grad = []
    for j in range(len(beta)):
        ones = np.where(matrix.T[j] == 1)
        theta_ones = np.exp(theta[ones])
        der_ones = np.sum(-np.exp(beta[j]) / (np.exp(beta[j]) + theta_ones))
        zeros = np.where(matrix.T[j] == 0)
        theta_zeros = np.exp(theta[zeros])
        der_zeros = np.sum(theta_zeros / (theta_zeros + np.exp(beta[j])))
        beta_grad.append(der_ones + der_zeros)
    beta += lr*np.array(beta_grad)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, sparse_matrix, val_data, lr, iterations, meta, s, weight):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param sparse_matrix: A sparse matrix representation of the data
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :param meta: subject metadata
    :param s: estimate for s
    :param weight: weight for Model A between 0-1 inclusive
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(sparse_matrix.get_shape()[0])
    beta = np.zeros(sparse_matrix.get_shape()[1])

    val_acc_lst = []
    num_iterations = [i for i in range(iterations)]
    train_lld = []
    val_lld = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(val_data, theta, beta, meta, s, weight)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(sparse_matrix, lr, theta, beta)
        train_lld.append(-neg_lld)
        val_lld.append(-neg_log_likelihood(val_data, theta, beta))

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, num_iterations, train_lld, val_lld


def evaluate(data, theta, beta, meta, s, weight):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param meta: subject metadata
    :param s: estimate for s
    :param weight: weight for Model A between 0-1 inclusive
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        # Uncomment for Part B
        # tot = 0
        # num = 0
        # for sub in meta[q]:
        #     if (s[u][0][sub] + s[u][1][sub]) > 0:
        #         num += 1
        #         tot += s[u][0][sub] / (s[u][0][sub] + s[u][1][sub])
        # avg = tot / num
        # p_a = p_a* weight + avg*(1-weight)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


# Function for Part B
def get_s(data):
    meta = {}
    # Iterate over the row to fill in the data.
    with open(os.path.join("../data", "question_meta.csv"), "r") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            try:
                subjects = row[1].strip('[').strip(']').split(',')
                meta[int(row[0])] = tuple(map(int, subjects))
            except ValueError:
                # Pass first row.
                pass
    s = {}
    for i in range(len(data['is_correct'])):
        student = data['user_id'][i]
        subjects = meta[data['question_id'][i]]
        if student not in s:
            s[student] = np.array([np.zeros(388), np.zeros(388)])
        for sub in subjects:
            if data['is_correct'][i] == 1:
                s[student][0][sub] += 1
            else:
                s[student][1][sub] += 1
    return meta, s


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    meta, s = get_s(train_data)
    theta, beta, val_acc_lst, num_iterations, train_lld, val_lld =\
        irt(train_data, sparse_matrix, val_data, 0.01, 25, meta, s, 1)
    plt.title('Avg Log-Likelihood During Gradient Ascent')
    plt.xlabel('Iteration')
    plt.ylabel('Avg Log-Likelihood')
    plt.plot(num_iterations, train_lld, label='Train')
    plt.plot(num_iterations, val_lld, label='Validation')
    plt.legend()
    plt.savefig('log_likelihoods.png')
    plt.clf()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (c)                                                #
    #####################################################################
    test_acc = evaluate(test_data, theta, beta, meta, s, 1)
    print("Final Validation Accuracy: {}".format(val_acc_lst[-1]))
    print("Final Test Accuracy: {}".format(test_acc))
    np.random.seed(5991)
    questions = np.random.choice(range(sparse_matrix.get_shape()[1]), 5, True)
    students = [i for i in range(sparse_matrix.get_shape()[0])]
    plt.title('Estimated Probabilities of Correct Answer for Each Student')
    plt.xlabel('Student ID')
    plt.ylabel('Est. Probability of Correct Answer')
    for ques in questions:
        diff = np.exp(theta - beta[ques])
        plt.plot(students, diff / (1+diff), label=ques)
    plt.legend(title='Ques. ID')
    plt.savefig('probabilities.png')
    plt.clf()

    # Part B
    # plt.title('Validation Accuracies for Different Weightings')
    # plt.xlabel('Iteration')
    # plt.ylabel('Accuracy')
    # for weight in [1, 0.75, 0.5, 0.25, 0]:
    #     print(weight)
    #     theta, beta, val_acc_lst, num_iterations, train_lld, val_lld =\
    #         irt(train_data, sparse_matrix, val_data, 0.01, 50, meta, s, weight)
    #     plt.plot(num_iterations, val_acc_lst, label=weight)
    # plt.legend(title='Weighting of Model A')
    # plt.savefig('val_accuracies.png')
    # plt.close()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
