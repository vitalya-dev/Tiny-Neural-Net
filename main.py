import numpy as np
import pandas as pd

import numpy as np
import struct

import json

def load_mnist_idx(train_images_path, train_labels_path, test_images_path, test_labels_path):
    def read_idx(filename):
        import struct
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
            data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.reshape(shape)

    X_train_raw = read_idx(train_images_path)
    Y_train = read_idx(train_labels_path)
    X_test_raw = read_idx(test_images_path)
    Y_test = read_idx(test_labels_path)

    # Меняем форму (reshape) под стандарты сверточных сетей:
    # m картинок, 1 цветовой канал, размер 28x28
    X_train = X_train_raw.reshape(X_train_raw.shape[0], 1, 28, 28)
    X_test = X_test_raw.reshape(X_test_raw.shape[0], 1, 28, 28)

    # Нормализуем данные (делаем значения пикселей от 0 до 1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return X_train, Y_train, X_test, Y_test

def init_params():
    # Слой 1: Сверточный слой (Convolutional Layer)
    # Создаем 8 фильтров. Каждый фильтр имеет глубину 1 (т.к. картинка черно-белая) и размер 3x3.
    # Форма: (количество_фильтров, количество_каналов, высота, ширина)
    # Мы умножаем случайные числа на 0.1, чтобы веса были маленькими — это важно для стабильного обучения.
    W1 = (np.random.rand(8, 1, 3, 3) - 0.5) * 0.1
    
    # Смещение (bias) для каждого из 8 фильтров.
    # Форма: (8 фильтров, 1 колонка, 1 строка)
    b1 = (np.random.rand(8, 1, 1) - 0.5) * 0.1
    
    # Слой 2: Выходной полносвязный слой (Dense Layer)
    # После свертки и пулинга у нас получится 8 "карт признаков" размером 13x13.
    # Мы вытягиваем их в линию: 8 * 13 * 13 = 1352 входа.
    # W2 соединяет эти 1352 входа с 10 возможными ответами (цифрами).
    W2 = (np.random.rand(10, 1352) - 0.5) * 0.1
    
    # Смещение для 10 выходных нейронов
    b2 = (np.random.rand(10, 1) - 0.5) * 0.1
    
    return W1, b1, W2, b2

def ReLU(Z):
    # np.maximum сравнивает каждый элемент матрицы с нулем
    # Если число меньше нуля, оно станет нулем.
    return np.maximum(Z, 0)

def deriv_ReLU(Z):
    # Возвращает True (1), если Z > 0, и False (0), если Z <= 0
    return Z > 0

def softmax(Z):
    # Возводим в экспоненту и делим на сумму по столбцам (axis=0).
    # Так мы получаем вероятности для каждой цифры.
    A = np.exp(Z) / sum(np.exp(Z))
    return A

def conv2d_forward(X, W, b):
    # X имеет форму (количество_картинок, каналы=1, высота=28, ширина=28)
    # W имеет форму (количество_фильтров=8, каналы=1, высота_фильтра=3, ширина_фильтра=3)
    m, c_prev, h_prev, w_prev = X.shape
    f_num, c_prev, f_h, f_w = W.shape
    
    # Вычисляем размер новой картинки. 
    # Если картинка 28x28, а фильтр 3x3, то он поместится 26 раз. Размер станет 26x26.
    h_out = h_prev - f_h + 1
    w_out = w_prev - f_w + 1
    
    # Создаем пустую матрицу для результата
    Z = np.zeros((m, f_num, h_out, w_out))
    
    # Скользим окном фильтра по высоте и ширине картинки
    for h in range(h_out):
        for w in range(w_out):
            # Вырезаем кусочек картинки размером 3x3 сразу для всех картинок (m)
            X_slice = X[:, :, h:h+f_h, w:w+f_w]
            
            # Умножаем вырезанный кусочек на наши фильтры и складываем результат.
            # np.tensordot быстро перемножает многомерные матрицы.
            Z[:, :, h, w] = np.tensordot(X_slice, W, axes=([1, 2, 3], [1, 2, 3]))
    
    # Добавляем смещение (bias) к каждому фильтру
    Z = Z + b.reshape(1, f_num, 1, 1)
    
    return Z

def maxpool2d_forward(X, size=2, stride=2):
    # Эта функция сжимает картинку в 2 раза, оставляя только самые яркие пиксели
    m, c, h_prev, w_prev = X.shape
    h_out = h_prev // size
    w_out = w_prev // size
    
    A = np.zeros((m, c, h_out, w_out))
    
    # Снова скользим окном, но теперь с шагом 2 (stride=2)
    for h in range(h_out):
        for w in range(w_out):
            h_start = h * stride
            w_start = w * stride
            # Вырезаем кусочек 2x2
            X_slice = X[:, :, h_start:h_start+size, w_start:w_start+size]
            # Находим максимум (самый яркий пиксель) в этом квадрате 2x2
            A[:, :, h, w] = np.max(X_slice, axis=(2, 3))
            
    return A

def maxpool2d_backward(dA_pool, A_conv, size=2, stride=2):
    # dA_pool - это ошибка, пришедшая от выходного слоя
    # A_conv - это картинки до сжатия (чтобы мы помнили, где были яркие пиксели)
    m, c, h_out, w_out = dA_pool.shape
    
    # Создаем пустую матрицу градиентов размером с исходную несжатую картинку
    dA_conv = np.zeros_like(A_conv)
    
    # Снова проходим окном по картинкам
    for h in range(h_out):
        for w in range(w_out):
            h_start = h * stride
            w_start = w * stride
            
            # Берем кусочек 2x2, который мы сжимали
            A_slice = A_conv[:, :, h_start:h_start+size, w_start:w_start+size]
            
            # Находим, какое значение в этом кусочке было максимальным
            max_vals = np.max(A_slice, axis=(2, 3), keepdims=True)
            
            # Создаем "маску" (True там, где был максимум, и False в остальных местах)
            mask = (A_slice == max_vals)
            
            # Передаем ошибку только победителям (умножаем маску на ошибку)
            dA_pool_slice = dA_pool[:, :, h, w].reshape(m, c, 1, 1)
            dA_conv[:, :, h_start:h_start+size, w_start:w_start+size] += mask * dA_pool_slice
            
    return dA_conv

def conv2d_backward(dZ_conv, X, W):
    # Эта функция вычисляет, как нужно изменить наши фильтры (dW)
    m, f_num, h_out, w_out = dZ_conv.shape
    f_num, c_prev, f_h, f_w = W.shape
    
    # Пустые матрицы для новых градиентов фильтров и смещений
    dW = np.zeros_like(W)
    db = np.sum(dZ_conv, axis=(0, 2, 3)).reshape(f_num, 1, 1)
    
    # Проходим по всем позициям, где был фильтр
    for h in range(h_out):
        for w in range(w_out):
            # Берем кусок исходной картинки
            X_slice = X[:, :, h:h+f_h, w:w+f_w]
            # Берем ошибку для этой позиции
            dZ_slice = dZ_conv[:, :, h, w]
            
            # Умножаем ошибку на картинку и добавляем к градиенту фильтра
            dW += np.tensordot(dZ_slice, X_slice, axes=([0], [0]))
            
    # Мы не считаем градиент для самой картинки X, потому что это входные данные, 
    # и дальше назад передавать ошибку некуда. Это экономит ресурсы!
    return dW, db

def one_hot(Y):
    # Создаем матрицу из нулей размером (количество картинок, 10 цифр)
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    # Ставим единицы в нужные столбцы, соответствующие правильным ответам
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Переворачиваем матрицу (транспонируем), чтобы столбцы стали картинками
    return one_hot_Y.T

def forward_prop(W1, b1, W2, b2, X):
    # 1. Слой свертки: ищем линии и узоры
    Z1_conv = conv2d_forward(X, W1, b1)
    
    # 2. Функция активации ReLU: убираем отрицательные числа
    A1_conv = ReLU(Z1_conv)
    
    # 3. Слой пулинга: сжимаем картинку 26x26 в 13x13 для ускорения
    A1_pool = maxpool2d_forward(A1_conv)
    
    # 4. Вытягивание (Flatten): превращаем 3D-карты признаков в плоскую линию.
    # A1_pool имеет форму (m, 8, 13, 13). 
    # Мы превращаем её в (1352, m), чтобы подать в классическую полносвязную сеть.
    m = X.shape[0]
    A1_flat = A1_pool.reshape(m, -1).T
    
    # 5. Выходной полносвязный слой: принимаем решение о цифре
    Z2 = W2.dot(A1_flat) + b2
    A2 = softmax(Z2)
    
    # Возвращаем все промежуточные результаты, они критически понадобятся для шага назад!
    return Z1_conv, A1_conv, A1_pool, A1_flat, Z2, A2

def backward_prop(Z1_conv, A1_conv, A1_pool, A1_flat, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    
    # 1. Шаг назад через выходной полносвязный слой (как было раньше)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1_flat.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    
    # 2. Передаем ошибку от полносвязного слоя обратно к слою пулинга.
    # Ошибка dZ2 имеет форму (10, m). Мы умножаем её на веса W2, чтобы получить ошибку для 1352 пикселей.
    dA1_flat = W2.T.dot(dZ2)
    
    # 3. Превращаем плоскую линию ошибки обратно в 3D-квадратики (разворачиваем Flatten)
    dA1_pool = dA1_flat.T.reshape(m, 8, 13, 13)
    
    # 4. Шаг назад через Max Pooling (передаем ошибку ярким пикселям)
    dA1_conv = maxpool2d_backward(dA1_pool, A1_conv)
    
    # 5. Шаг назад через ReLU (отключаем градиент там, где значения были меньше нуля)
    dZ1_conv = dA1_conv * deriv_ReLU(Z1_conv)
    
    # 6. Шаг назад через Свертку (считаем градиенты для наших фильтров 3x3)
    dW1, db1 = conv2d_backward(dZ1_conv, X, W1)
    
    # Усредняем градиенты по количеству картинок в партии
    dW1 = dW1 / m
    db1 = db1 / m
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # Обновляем параметры, вычитая градиент, умноженный на скорость обучения (alpha)
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    
    # Возвращаем новые, немного улучшенные веса и смещения
    return W1, b1, W2, b2

def gradient_descent(X, Y, alpha, iterations):
    # Инициализируем новые фильтры и веса
    W1, b1, W2, b2 = init_params()
    
    for i in range(iterations):
        # 1. Прямой проход (принимаем 6 переменных от новых слоев)
        Z1_conv, A1_conv, A1_pool, A1_flat, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        
        # 2. Обратный проход (прокидываем ошибку через пулинг и фильтры)
        dW1, db1, dW2, db2 = backward_prop(Z1_conv, A1_conv, A1_pool, A1_flat, Z2, A2, W1, W2, X, Y)
        
        # 3. Обновление весов (функция остается без изменений)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        if i % 10 == 0:
            print(f"Итерация: {i}")
            predictions = get_predictions(A2)
            print(f"Точность на обучающей выборке: {get_accuracy(predictions, Y):.4f}")
            
    return W1, b1, W2, b2

def get_predictions(A2):
    # Функция argmax находит индекс самого большого числа в столбце.
    # Поскольку индексы совпадают с цифрами (0-9), это и есть предсказанная цифра.
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # Сравниваем массивы предсказаний и правильных ответов (получаем значения True/False).
    # Суммируем совпадения и делим на общее число картинок.
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2):
    # При предсказании нам нужен только финальный ответ A2, 
    # остальные промежуточные данные игнорируем с помощью символа "_"
    _, _, _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

if __name__ == "__main__":
    print("Загрузка данных...")
    X_train, Y_train, X_test, Y_test = load_mnist_idx(
        'train-images-idx3-ubyte', 'train-labels-idx1-ubyte', 
        't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'
    )
    
    # БЕРЕМ ТОЛЬКО ЧАСТЬ ДАННЫХ ДЛЯ ТЕСТА (иначе цикл будет идти слишком долго)
    X_train_small = X_train[:1000]
    Y_train_small = Y_train[:1000]
    X_test_small = X_test[:200]
    Y_test_small = Y_test[:200]
    
    print("Данные готовы. Начинаем обучение CNN (это займет время)...")
    # Запускаем на 50 итераций с шагом 0.1
    W1, b1, W2, b2 = gradient_descent(X_train_small, Y_train_small, 0.1, 50)
    
    print("---")
    print("Тестирование на новых данных...")
    test_predictions = make_predictions(X_test_small, W1, b1, W2, b2)
    print(f"Точность на тестовой выборке: {get_accuracy(test_predictions, Y_test_small) * 100:.2f}%")
