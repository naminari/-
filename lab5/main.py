import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util as u

# Загрузка данных
dataset = pd.read_csv('./mushroom.csv')

# Разделение на признаки и целевую переменную
X = dataset.drop('poisonous', axis=1)
y = dataset['poisonous']

# Выводим первые строки данных
print(X.head())
print("------")
print(y.head())

# Преобразуем категориальные признаки в числовые
X_selected = pd.get_dummies(X, columns=X.columns)

# Печатаем первые строки преобразованных данных
print(X_selected.head())

# Преобразование y в числовые значения
y_numeric = (y == 'p').astype(int)

# Разделение на обучающую и тестовую выборки
np.random.seed(42)
indices = np.random.permutation(len(X_selected))
split_index = int(0.8 * len(X_selected))
X_train, X_test = X_selected.values[indices[:split_index]], X_selected.values[indices[split_index:]]
y_train, y_test = y_numeric.values[indices[:split_index]], y_numeric.values[indices[split_index:]]

def print_tree(tree, indent=""):
    """Вывод структуры дерева в текстовом формате."""
    if tree.value is not None:
        print(indent + "Класс:", tree.value)
    else:
        # Выводим текущий признак
        print(indent + f"Признак {X_selected.columns[tree.feature]}")

        # Для каждого уникального значения признака выводим ветвь
        for branch_value, subtree in tree.branches.items():
            print(indent + f"--> Значение {branch_value}:")
            print_tree(subtree, indent + "    ")

# Строим дерево
tree = u.build_tree(X_train, y_train, max_depth=3)

# Печать дерева
print_tree(tree)

# Прогнозируем и вычисляем метрики
y_pred = [u.predict(tree, x) for x in X_test]
y_pred = np.array(y_pred)

y_probs = [u.predict_proba(tree, x) for x in X_test]
y_probs = np.array(y_probs)  # Преобразуем в NumPy array

# Вывод уникальных значений вероятностей
print("Уникальные вероятности:", np.unique(y_probs))

# Расчет точек для ROC и PR кривых
fpr, tpr = u.calculate_roc_points(y_test, y_probs)
precision, recall = u.calculate_pr_points(y_test, y_probs)

# Убедимся, что точки отсортированы
fpr, tpr = zip(*sorted(zip(fpr, tpr)))


# Расчет AUC для ROC и PR кривых
auc_roc = u.auc(fpr, tpr)

sorted_indices = np.argsort(recall)
recall_sorted = np.array(recall)[sorted_indices]
precision_sorted = np.array(precision)[sorted_indices]

auc_pr = np.trapezoid(precision_sorted, recall_sorted)


print(f"AUC ROC: {auc_roc:.4f}")
print(f"AUC PR: {auc_pr:.4f}")

print("Точки ROC:", list(zip(fpr, tpr)))  # Выводим точки для ROC
print("Точки PR:", list(zip(recall, precision)))  # Выводим точки для PR


