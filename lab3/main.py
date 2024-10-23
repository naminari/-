import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import random
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def knn_predict(X_train, y_train, X_test, k=5):
    predictions = []
    
    for test_point in X_test:
        distances = []
        for i, train_point in enumerate(X_train):
            dist = euclidean_distance(test_point, train_point)
            distances.append((dist, y_train[i]))
        
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = [neighbor[1] for neighbor in distances[:k]]
        
        most_common = Counter(k_nearest_neighbors).most_common(1)[0][0]
        predictions.append(most_common)
    
    return predictions

DATASET_PATH = r'/home/naminari/Загрузки/Telegram Desktop/asi/lab3/WineDataset.csv'
df = pd.read_csv(DATASET_PATH)

x = df['Alcohol']
y = df['Color intensity']
z = df['Proline']

x_25, x_50, x_75 = x.quantile(0.25), x.quantile(0.50), x.quantile(0.75)
y_25, y_50, y_75 = y.quantile(0.25), y.quantile(0.50), y.quantile(0.75)
z_25, z_50, z_75 = z.quantile(0.25), z.quantile(0.50), z.quantile(0.75)

x_min, x_max, x_mean = x.min(), x.max(), x.mean()
y_min, y_max, y_mean = y.min(), y.max(), y.mean()
z_min, z_max, z_mean = z.min(), z.max(), z.mean()

print(f"Алкоголь: Мин: {x_min}, Макс: {x_max}, Среднее: {x_mean}, 25% квантиль: {x_25}, Медиана: {x_50}, 75% квантиль: {x_75}")
print(f"Цветовая интенсивность: Мин: {y_min}, Макс: {y_max}, Среднее: {y_mean}, 25% квантиль: {y_25}, Медиана: {y_50}, 75% квантиль: {y_75}")
print(f"Пролин: Мин: {z_min}, Макс: {z_max}, Среднее: {z_mean}, 25% квантиль: {z_25}, Медиана: {z_50}, 75% квантиль: {z_75}")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z, label="Точки данных")

ax.scatter(x_min, y_min, z_min, color='red', s=100, label='Минимум')
ax.text(x_min, y_min, z_min, 'Минимум', color='red')

ax.scatter(x_max, y_max, z_max, color='green', s=100, label='Максимум')

ax.scatter(x_mean, y_mean, z_mean, color='blue', s=100, label='Среднее')

ax.scatter(x_25, y_25, z_25, color='orange', s=100, label='25% Квантиль')
ax.scatter(x_50, y_50, z_50, color='purple', s=100, label='50% Квантиль (Медиана)')
ax.scatter(x_75, y_75, z_75, color='cyan', s=100, label='75% Квантиль')

ax.set_xlabel('Алкоголь')
ax.set_ylabel('Цветовая интенсивность')
ax.set_zlabel('Пролин')
ax.legend()
plt.show()

scaler = StandardScaler()
scaler.fit(df.drop('Wine', axis=1))
scaled_features = scaler.transform(df.drop('Wine', axis=1))
scaled_data = pd.DataFrame(scaled_features, columns=df.drop('Wine', axis=1).columns)

print(df.info())

x = scaled_data
y = df['Wine']

x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size=0.3)

# Сброс индексов после разделения
y_training_data = y_training_data.reset_index(drop=True)
y_test_data = y_test_data.reset_index(drop=True)

def evaluate_model(X_train, y_train, X_test, y_test, k_values):
    for k in k_values:
        print(f"Оценка модели при k={k}")
        y_pred = knn_predict(X_train, y_train, X_test, k)
        
        accuracy = np.mean(y_pred == y_test)
        print(f"Точность: {accuracy}")

        print("Матрица неточностей:\n", confusion_matrix(y_test, y_pred))
        print("Отчет классификации:\n", classification_report(y_test, y_pred))

random.seed(42)
random_features = random.sample(list(x.columns), k=3)
print(f"Случайно выбранные признаки для Модели 1: {random_features}")

X_train_random = x_training_data[random_features].values
X_test_random = x_test_data[random_features].values

fixed_features = ['Alcohol', 'Color intensity', 'Proline']
print(f"Фиксированные признаки для Модели 2: {fixed_features}")

X_train_fixed = x_training_data[fixed_features].values
X_test_fixed = x_test_data[fixed_features].values

k_values = [3, 5, 10]

print("\nОценка Модели 1 (Случайные признаки):")
evaluate_model(X_train_random, y_training_data, X_test_random, y_test_data, k_values)

print("\nОценка Модели 2 (Фиксированные признаки):")
evaluate_model(X_train_fixed, y_training_data, X_test_fixed, y_test_data, k_values)
