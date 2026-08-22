import numpy as np
import pandas as pd

import numpy as np
import struct

def load_mnist_idx(train_images_path, train_labels_path, test_images_path, test_labels_path):
    # Вспомогательная функция для чтения одного IDX файла
    def read_idx(filename):
        with open(filename, 'rb') as f:
            # Читаем первые 4 байта (magic number)
            # '>HBB' означает: Big-endian, H (2 байта нулей), B (1 байт типа данных), B (1 байт размерности)
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            
            # Читаем размеры для каждого измерения (каждый размер - это 4-байтовое число '>I')
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            
            # Читаем оставшиеся данные как одномерный массив (тип 0x08 - это беззнаковый байт uint8)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            
            # Превращаем одномерный массив в матрицу нужной формы
            return data.reshape(shape)

    # 1. Считываем сырые данные из 4-х файлов
    X_train_raw = read_idx(train_images_path)
    Y_train = read_idx(train_labels_path)
    X_test_raw = read_idx(test_images_path)
    Y_test = read_idx(test_labels_path)

    # 2. Подготавливаем матрицы для нейросети
    # Исходная форма X_train_raw: (60000, 28, 28). 
    # Нам нужно вытянуть картинки в столбцы, чтобы форма стала (784, 60000)
    X_train = X_train_raw.reshape(X_train_raw.shape[0], -1).T
    X_test = X_test_raw.reshape(X_test_raw.shape[0], -1).T

    # 3. Нормализуем данные (делаем значения пикселей от 0 до 1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, Y_train, X_test, Y_test

def init_params():
    # W1 - матрица весов от входа (784) к скрытому слою (10 нейронов)
    # np.random.rand генерирует числа от 0 до 1. 
    # Вычитаем 0.5, чтобы получить значения от -0.5 до 0.5
    W1 = np.random.rand(10, 784) - 0.5
    
    # b1 - вектор смещений для скрытого слоя (10 строк, 1 колонка)
    b1 = np.random.rand(10, 1) - 0.5
    
    # W2 - матрица весов от скрытого слоя (10) к выходному (10 цифр)
    W2 = np.random.rand(10, 10) - 0.5
    
    # b2 - вектор смещений для выходного слоя (10 строк, 1 колонка)
    b2 = np.random.rand(10, 1) - 0.5
    
    return W1, b1, W2, b2

def ReLU(Z):
    # np.maximum сравнивает каждый элемент матрицы с нулем
    # Если число меньше нуля, оно станет нулем.
    return np.maximum(Z, 0)

def softmax(Z):
    # Возводим в экспоненту и делим на сумму по столбцам (axis=0).
    # Так мы получаем вероятности для каждой цифры.
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def forward_prop(W1, b1, W2, b2, X):
    # 1. Считаем сырые значения для скрытого слоя (умножение матриц .dot)
    Z1 = W1.dot(X) + b1
    # 2. Пропускаем через функцию активации ReLU
    A1 = ReLU(Z1)
    
    # 3. Считаем сырые значения для выходного слоя
    Z2 = W2.dot(A1) + b2
    # 4. Превращаем в вероятности с помощью Softmax
    A2 = softmax(Z2)
    
    # Нам понадобятся эти значения позже для обучения, поэтому возвращаем их все
    return Z1, A1, Z2, A2

def one_hot(Y):
    # Создаем матрицу из нулей размером (количество картинок, 10 цифр)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # Ставим единицы в нужные столбцы, соответствующие правильным ответам
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Переворачиваем матрицу (транспонируем), чтобы столбцы стали картинками
    return one_hot_Y.T

def deriv_ReLU(Z):
    # Возвращает True (1), если Z > 0, и False (0), если Z <= 0
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    # m - количество картинок в нашем наборе
    m = Y.size
    
    # Превращаем правильные ответы в формат из 0 и 1
    one_hot_Y = one_hot(Y)
    
    # Считаем ошибку на выходном слое (Предсказание минус правильный ответ)
    dZ2 = A2 - one_hot_Y
    # Считаем, как нужно изменить веса и смещения второго слоя
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    # Переносим ошибку на скрытый слой, умножая на веса W2 (в обратную сторону)
    # и применяем производную от функции активации ReLU
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    # Считаем, как нужно изменить веса и смещения первого слоя
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    
    # Возвращаем градиенты (направления и силу изменения весов)
    return dW1, db1, dW2, db2

def main():
    X_train, Y_train, X_test, Y_test = load_mnist_idx(
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte"
    )

    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_test: ", X_test.shape)
    print("Y_test: ", Y_test.shape)

    print()
    print("Первая метка:", Y_train[0])
    print("Первые 10 пикселей:", X_train[:10, 0])
    print("Минимальный пиксель:", X_train.min())
    print("Максимальный пиксель:", X_train.max())


if __name__ == "__main__":
    main()