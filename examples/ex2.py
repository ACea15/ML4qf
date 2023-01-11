import ml4qf.inputs
import ml4qf.collectors.financial_features as financial_features

#https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html#sphx-glr-auto-examples-tree-plot-iris-dtc-py

from sklearn.datasets import load_iris

iris = load_iris()

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.svm import SVC
# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02


for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier().fit(X, y)
    #clf = SVC().fit(X, y)

    # Plot the decision boundary
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]],
    )

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0],
            X[idx, 1],
            c=color,
            label=iris.target_names[i],
            cmap=plt.cm.RdYlBu,
            edgecolor="black",
            s=15,
        )

plt.suptitle("Decision surface of decision trees trained on pairs of features")
plt.legend(loc="lower right", borderpad=0, handletextpad=0)
_ = plt.axis("tight")
plt.show()

from sklearn.tree import plot_tree

plt.figure()
clf = DecisionTreeClassifier().fit(iris.data, iris.target)
plot_tree(clf, filled=True)
plt.title("Decision tree trained on all the iris features")
plt.show()


data = iris.data
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)
labels = iris.target


def classify(som, data):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result



########
from minisom import MiniSom
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels)

som = MiniSom(7, 7, 4, sigma=3, learning_rate=0.5, 
              neighborhood_function='triangle', random_seed=10)
som.pca_weights_init(X_train)
som.train_random(X_train, 500, verbose=False)

print(classification_report(y_test, classify(som, X_test)))

clf = DecisionTreeClassifier().fit(X_train, y_train)
print(classification_report(y_test, clf.predict( X_test)))

########
import umap

# embedding = umap.UMAP(
#     n_neighbors=30,
#     min_dist=0.0,
#     n_components=2,
#     random_state=42,
# ).fit_transform(X_train, y_train)

trans = umap.UMAP(n_neighbors=15, n_components=3, random_state=42).fit(X_train)
dtc = DecisionTreeClassifier().fit(trans.embedding_, y_train)
test_embedding = trans.transform(X_test)
print(classification_report(y_test, dtc.predict(test_embedding)))



################

# from sklearn.pipeline import Pipeline

# pipe = Pipeline([('embedding', umap.UMAP(n_neighbors=15, n_components=3, random_state=42)),
#                  ('dtc', DecisionTreeClassifier())])
# pipe.fit(X_train, y_train)
# print(classification_report(y_test, pipe.predict(X_test)))

# import ml4qf.inputs as In

######################
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator 
import ml4qf.inputs as In
from ml4qf.predictors import model_keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

iris = load_iris()
data = iris.data
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)
labels = iris.target
X_train, X_test, y_train, y_test = train_test_split(data, labels, stratify=labels)
num_features = X_train.shape[1]
seqlen = 5 
#X_testkeras = TimeseriesGenerator(X_test, y_test, length=seqlen)
#X_trainkeras = TimeseriesGenerator(X_train, y_train, length=seqlen)

train_shape =  X_train.shape
test_shape =  X_test.shape

X_trainkeras = X_train.reshape((train_shape[0], 1, train_shape[1]))
X_testkeras = X_test.reshape((test_shape[0], 1, test_shape[1]))

layers_dict = dict()
layers_dict['LSTM'] = dict(units=5, input_shape=(1, train_shape[1]),
                           activation = 'relu', return_sequences=False, name='LSTM')
layers_dict['Dense'] = dict(units=1, name='Output')

Ilstm = In.Input_keras(layers_dict)
lstm = model_keras.Model_keras(Ilstm.KERAS_MODEL, Ilstm.LAYERS)
lstm.fit(X_trainkeras, y_train)

print(classification_report(y_test, lstm._model.predict(X_testkeras)))




