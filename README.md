Сперва импортируем необходимые библеотеки:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import winsound

import scipy.sparse as sparse

from lightfm import LightFM
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
from lightfm.data import Dataset

import sklearn
from sklearn.model_selection import train_test_split
```
Далее загружаем необходимые нам датасеты:
```ptyhon
data = pd.read_csv('data/train.csv', low_memory=False)
test = pd.read_csv('data/test.csv', low_memory=False)
submission = pd.read_csv('data/sample_submission.csv')
```
Используем метод grid search для нахождения наилучших параметров для нашей модели:
```python
def sample_hyperparameters():

    while True:
        yield {
            "no_components": np.random.randint(60, 120),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "num_epochs": np.random.randint(5, 40),
        }


def random_search(train, test, num_samples=10, num_threads=1):
    """
    Sample random hyperparameters, fit a LightFM model, and evaluate it
    on the test set.

    Parameters
    ----------

    train: np.float32 coo_matrix of shape [n_users, n_items]
        Training data.
    test: np.float32 coo_matrix of shape [n_users, n_items]
        Test data.
    num_samples: int, optional
        Number of hyperparameter choices to evaluate.


    Returns
    -------

    generator of (auc_score, hyperparameter dict, fitted model)

    """

    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train, epochs=num_epochs, num_threads=num_threads)
        preds = model.predict(test.userid.values,
                      test.itemid.values)
        score = sklearn.metrics.roc_auc_score(test.rating,preds)

        hyperparams["num_epochs"] = num_epochs

        yield (score, hyperparams, model)


if __name__ == "__main__":
    data = pd.read_csv('data/train.csv', low_memory=False)
    test = pd.read_csv('data/test.csv', low_memory=False)
    train_data, test_data = train_test_split(data,random_state=32, shuffle=True)
    ratings_coo = sparse.coo_matrix((train_data['rating'].astype(int),
                                 (train_data['userid'],
                                  train_data['itemid'])))
    (score, hyperparams, model) = max(random_search(ratings_coo, test_data, num_threads=4, num_samples=200), key=lambda x: x[0])
    winsound.Beep(freq, duration)
    print("Best score {} at {}".format(score, hyperparams))
```
