import warnings

from fontTools.misc.cython import returns

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = r'/home/naminari/Загрузки/Telegram Desktop/asi/lab4/california_housing_train.csv'
df = pd.read_csv(DATASET_PATH)

longitude_array = df['longitude'].values                              # долгота
latitude_array = df['latitude'].values                                # широта
avg_age_array = df['housing_median_age'].values                       # средний возраст жителей
total_rooms_array = df['total_rooms'].values                          # общее количество комнат
total_bedrooms_array = df['total_bedrooms'].values                    # общее количество спален
population_array = df['population'].values                            # население
households_array = df['households'].values                            # количество домохозяйств
income_array = df['median_income'].values                             # средний доход
housing_cost_array = df['median_house_value'].values                  # средняя стоимость жилья

# Индекс доступности жилья
df['HAI'] = (df['median_income'] / df['median_house_value']) * 100    # Индекс доступности жилья

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
def min_max_normalize(df):
    return (df - df.min()) / (df.max() - df.min())


def train_test_split(X, y=None, test_size=0.25, shuffle=True, random_state=None):
    X = np.array(X)
    if y is not None:
        y = np.array(y)
    if isinstance(test_size, float):
        test_size = int(len(X) * test_size)
    indices = np.arange(len(X))
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]

    if y is not None:
        y_train, y_test = y[train_indices], y[test_indices]
        return X_train, X_test, y_train, y_test

    return X_train, X_test

X = min_max_normalize(df.drop(columns=['HAI']))
y = df['HAI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X.describe(percentiles=[.25, .5, .75, .95]))


X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

def perform_linear_regression(columns, X_train, X_test, y_train, y_test):
    if columns:
        X_train = X_train[columns]
        X_test = X_test[columns]

    # Добавляем столбец с единицами для свободного члена
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])

    y_train = y_train.values if hasattr(y_train, 'values') else y_train
    y_test = y_test.values if hasattr(y_test, 'values') else y_test

    coefficients = np.linalg.lstsq(X_train, y_train, rcond=None)[0]

    def r2_score_custom(y_true, y_pred):
        total_variance = np.sum((y_true - np.mean(y_true))**2)
        residual_variance = np.sum((y_true - y_pred)**2)
        return 1 - (residual_variance / total_variance)

    def sum_of_squares(y_true, y_pred):
        return np.sum(np.square(y_true - y_pred))

    y_pred = np.dot(X_test, coefficients)

    r2 = r2_score_custom(y_test, y_pred)
    sum_of_squares_value = sum_of_squares(y_test, y_pred)

    return y_pred, r2, sum_of_squares_value

y_pred, r2, sum_of_squares = perform_linear_regression(['longitude', 'latitude'], X_train, X_test, y_train, y_test)
print('Коэффициент детерминации:', r2)
print('Предсказания:', y_pred)
print('Сумма квадратов:', sum_of_squares)

y_pred, r2, sum_of_squares = perform_linear_regression(['median_income', 'total_rooms'], X_train, X_test, y_train, y_test)
print('Коэффициент детерминации (R^2):', r2)
print('Предсказания:', y_pred)
print('Сумма квадратов:', sum_of_squares)

y_pred, r2, sum_of_squares = perform_linear_regression(list(X.columns), X_train, X_test, y_train, y_test)
print('Коэффициент детерминации:', r2)
print('Предсказания:', y_pred)
print('Сумма квадратов:', sum_of_squares)
