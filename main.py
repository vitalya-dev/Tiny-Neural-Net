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