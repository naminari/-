import warnings
import utils as util

from fontTools.misc.cython import returns

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = r'diabetes.csv'
df = pd.read_csv(DATASET_PATH)

# Вывод статистики
print(df.info())
print(df.describe(percentiles=[.25, .5, .75, .95]))

df.hist(bins=60, figsize=(20, 10))
plt.show()

plt.rcParams['figure.figsize'] = [5, 5]

data_count = df.count()
data_count.plot(kind='bar', title='Количество')
plt.show()

data_means = df.mean()
data_means.plot(kind='bar', title='Средние значения признаков')
plt.show()

data_std = df.std()
data_std.plot(kind='bar', title='Стандартные отклонения признаков')
plt.show()

data_min = df.min()
data_min.plot(kind='bar', title='Минимум')
plt.show()

data_max = df.max()
data_max.plot(kind='bar', title='Максимум')
plt.show()

plt.matshow(df.corr())
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.show()

df.corr().style.background_gradient(cmap='coolwarm').format(precision=2)

# Обработка отсутствующих значений (замена NaN средними значениями)
df.fillna(df.mean(), inplace=True)

# Категориальных признаков не выявлено

# Нормировка данных (приведение к единому масштабу)
def standardize_data(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    return standardized_data

X = standardize_data(df.drop(columns=['Outcome']))
y = df['Outcome']


X_train, X_test, y_train, y_test = util.train_test_split(X, y, test_size=0.2)

best_params = util.hyperparameter_search(X_train, y_train, X_test, y_test)
    
    # Обучение модели с лучшими гиперпараметрами
model = util.GDLogisticRegression(learning_rate=best_params['learning_rate'], max_iter=best_params['max_iter'])
model.fit(X_train, y_train)
pred_res = model.predict(X_test)
    
    # Оценка модели
accuracy = util.accuracy_score_manual(y_test, pred_res)
precision = util.precision_score_manual(y_test, pred_res)
recall = util.recall_score_manual(y_test, pred_res)
f1 = util.f1_score_manual(y_test, pred_res)
    
print(f'Accuracy:  {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}')

