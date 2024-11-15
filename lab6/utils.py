import numpy as np

class GDLogisticRegression:
    def __init__(self, learning_rate=0.1, tolerance=0.0001, max_iter=1000):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _log_loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.bias = 0
        self.weights = np.zeros(n_features)

        for i in range(self.max_iter):
            # Предсказание и вычисление функции потерь
            y_pred = self._sigmoid(X @ self.weights + self.bias)
            loss = self._log_loss(y, y_pred)

            # Вычисление градиентов
            db = np.mean(y_pred - y)
            dw = X.T @ (y_pred - y) / n_samples

            # Обновление весов и смещения
            self.bias -= self.learning_rate * db
            self.weights -= self.learning_rate * dw

            # Проверка на критерий остановки
            if np.linalg.norm(dw) < self.tolerance:
                break

    def predict(self, X):
        y_pred = self._sigmoid(X @ self.weights + self.bias)
        return (y_pred >= 0.5).astype(int)

    
def accuracy_score_manual(y_true, y_pred):
    """
    Функция для вычисления accuracy (точности) классификации.
    
    Параметры:
    y_true (array-like): Истинные метки классов
    y_pred (array-like): Предсказанные метки классов
    
    Возвращает:
    float: Точность предсказаний (accuracy)
    """
    # Преобразуем в numpy массивы для удобства работы
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Сравниваем предсказанные значения с истинными и считаем долю верных предсказаний
    correct_predictions = np.sum(y_true == y_pred)
    accuracy = correct_predictions / len(y_true)
    
    return accuracy

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

def hyperparameter_search(X_train, y_train, X_test, y_test):
    learning_rates = [0.01, 0.05, 0.1]
    max_iters = [500, 1000, 2000]
    
    best_score = 0
    best_params = {}
    
    for lr in learning_rates:
        for max_iter in max_iters:
            model = GDLogisticRegression(learning_rate=lr, max_iter=max_iter)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            score = accuracy_score_manual(y_test, predictions)
            
            print(f"Accuracy with learning rate={lr}, max_iter={max_iter}: {score}")
            
            if score > best_score:
                best_score = score
                best_params = {'learning_rate': lr, 'max_iter': max_iter}
                
    print(f"Best parameters: {best_params}")
    return best_params

def precision_score_manual(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive)

def recall_score_manual(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative)

def f1_score_manual(y_true, y_pred):
    precision = precision_score_manual(y_true, y_pred)
    recall = recall_score_manual(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)
