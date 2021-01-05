from q2.check_grad import check_grad
from q2.utils import *
from q2.logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    hyperparameters = {
        "learning_rate": 0.01,
        "weight_regularization": 0.,
        "num_iterations": 1000
    }
    zeros = []
    for _ in range(M+1):
        zeros.append([0])
    weights = np.array(zeros)

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    train_ces = []
    valid_ces = []
    for t in range(hyperparameters["num_iterations"]):
        descent = np.multiply(hyperparameters["learning_rate"],
                              logistic(weights, train_inputs, train_targets,
                                       hyperparameters)[1])
        weights = np.subtract(weights, descent)
        y = logistic_predict(weights, train_inputs)
        train_ces.append(evaluate(train_targets, y)[0])
        y = logistic_predict(weights, valid_inputs)
        valid_ces.append(evaluate(valid_targets, y)[0])
    y = logistic_predict(weights, test_inputs)
    print(evaluate(test_targets, y))
    plt.title('Cross Entropy as Training Progresses (mnist_train_small)')
    plt.xlabel('Iteration')
    plt.ylabel('Cross Entropy')
    plt.plot(range(hyperparameters["num_iterations"]), train_ces, label='Train')
    plt.plot(range(hyperparameters["num_iterations"]), valid_ces, label='Valid')
    plt.legend()
    plt.show()


def run_pen_logistic_regression():
    train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()

    N, M = train_inputs.shape

    hyperparameters = {
        "learning_rate": 0.01,
        "weight_regularization": 0.,
        "num_iterations": 1500
    }
    zeros = []
    for _ in range(M+1):
        zeros.append([0])
    weights = np.array(zeros)
    for lamb in [0, 0.001, 0.01, 0.1, 1]:
        hyperparameters["weight_regularization"] = lamb
        train_ces = []
        valid_ces = []
        for t in range(hyperparameters["num_iterations"]):
            descent = np.multiply(hyperparameters["learning_rate"],
                                  logistic_pen(weights, train_inputs,
                                               train_targets,
                                               hyperparameters)[1])
            weights = np.subtract(weights, descent)
            y = logistic_predict(weights, train_inputs)
            train_ces.append(evaluate(train_targets, y)[0])
            y = logistic_predict(weights, valid_inputs)
            valid_ces.append(evaluate(valid_targets, y)[0])
        plt.title('Cross Entropy as Training Progresses(mnist_train_small),'
                  'lambda=1.0')
        plt.xlabel('Iteration')
        plt.ylabel('Cross Entropy')
        plt.plot(range(hyperparameters["num_iterations"]), train_ces,
                 label='Train')
        plt.plot(range(hyperparameters["num_iterations"]), valid_ces,
                 label='Valid')
        plt.legend()
        plt.show()
        ces = []
        class_errors = []
        for _ in range(5):
            f, df, y = logistic_pen(weights, test_inputs, test_targets,
                                    hyperparameters)
            ces.append(f)
            class_errors.append(evaluate(test_targets, y)[1])
        print(np.mean(ces), np.mean(class_errors))


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    # run_pen_logistic_regression()
