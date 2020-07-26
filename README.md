Сперва импортируем необходимые библеотеки:
'''python
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
'''
Далее загружаем необходимые нам датасеты:
'''ptyhon
data = pd.read_csv('data/train.csv', low_memory=False)
test = pd.read_csv('data/test.csv', low_memory=False)
submission = pd.read_csv('data/sample_submission.csv')
'''
