import numpy as np
import matplotlib.pyplot as plt


data_train = {'X': np.genfromtxt('data_train_X.csv', delimiter=','),
              't': np.genfromtxt('data_train_y.csv', delimiter=',')}
data_test = {'X': np.genfromtxt('data_test_X.csv', delimiter=','),
             't': np.genfromtxt('data_test_y.csv', delimiter=',')}


def shuffle_data(data):
    t = data[0]
    X = data[1]
    np.random.seed(5991)
    p = np.random.permutation(len(t))
    return t[p], X[p]


def split_data(data, num_folds, fold):
    data_fold = []
    data_rest = []
    for arr in data:
        new = np.array_split(arr, num_folds)
        saved = new.pop(fold)
        data_fold.append(saved)
        combined = np.concatenate(new)
        data_rest.append(combined)
    return tuple(data_fold), tuple(data_rest)


def train_model(data, lambd):
    t = data[0]
    X = data[1]
    diagonal = np.diag([lambd*len(t)]*len(np.transpose(X)))
    sum = np.add(np.matmul(np.transpose(X), X), diagonal)
    inverse = np.linalg.inv(sum)
    product = np.matmul(inverse, np.transpose(X))
    return product.dot(t)


def predict(data, model):
    X = data[1]
    return X.dot(model)


def loss(data, model):
    t = data[0]
    prediction = predict(data, model)
    error_loss = np.subtract(prediction, t)
    return error_loss.dot(error_loss) / (2*len(t))


def cross_validation(data, num_folds, lambd_seq):
    cv_error = [0]*len(lambd_seq)
    data = shuffle_data(data)
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(num_folds):
            val_cv, train_cv = split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
        cv_error[i] = cv_loss_lmd / num_folds
    return cv_error


data_train = (data_train['t'], data_train['X'])
data_test = (data_test['t'], data_test['X'])
lambd_seq = np.linspace(0.00005, 0.005)
train_errors = []
test_errors = []
for lambd in lambd_seq:
    model = train_model(data_train, lambd)
    train_errors.append(loss(data_train, model))
    test_errors.append(loss(data_test, model))
plt.title('Errors for Different Values of Lambda')
plt.xlabel('Lambda')
plt.ylabel('Error')
plt.plot(lambd_seq, train_errors, label='Train Errors')
plt.plot(lambd_seq, test_errors, label='Test Errors')
plt.plot(lambd_seq, cross_validation(data_train, 5, lambd_seq),
         label='Cross Val. with 5')
plt.plot(lambd_seq, cross_validation(data_train, 10, lambd_seq),
         label='Cross Val. with 10')
plt.legend()
plt.show()
