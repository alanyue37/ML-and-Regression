from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def load_data():
    real_file = open('clean_real.txt', 'r')
    data = real_file.read().splitlines()
    real_file.close()
    num_real = len(data)
    labels = [1]*num_real
    fake_file = open('clean_fake.txt', 'r')
    data += fake_file.read().splitlines()
    fake_file.close()
    num_fake = len(data) - num_real
    labels.extend([0]*num_fake)
    train, rest, train_labels, rest_labels =\
        train_test_split(data, labels, test_size=0.3, random_state=5991)
    test, validate, test_labels, validate_labels =\
        train_test_split(rest, rest_labels, test_size=0.5, random_state=5991)
    design_matrix = CountVectorizer()
    train = design_matrix.fit_transform(train)
    test = design_matrix.transform(test)
    validate = design_matrix.transform(validate)
    return [train, train_labels, test, test_labels, validate, validate_labels]


def select_knn_model(train, train_labels, test, test_labels, validate,
                     validate_labels):
    scores_train = []
    scores_validate = []
    for k in range(20):
        neigh = KNeighborsClassifier(n_neighbors=k+1, metric='cosine')
        neigh.fit(train, train_labels)
        scores_train.append(neigh.score(train, train_labels))
        scores_validate.append(neigh.score(test, test_labels))
    new = [i for i in range(1, 21)]
    plt.title('Accuracy for Different k (metric=cosine)')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.plot(new, scores_train, label='Train')
    plt.plot(new, scores_validate, label='Test')
    plt.legend()
    plt.show()
    neigh = KNeighborsClassifier(n_neighbors=4)
    neigh.fit(train, train_labels)
    print(neigh.score(validate, validate_labels))


data = load_data()
select_knn_model(data[0], data[1], data[2], data[3], data[4], data[5])
