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

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # Обновляем параметры, вычитая градиент, умноженный на скорость обучения (alpha)
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    
    # Возвращаем новые, немного улучшенные веса и смещения
    return W1, b1, W2, b2

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

def get_predictions(A2):
    # Функция argmax находит индекс самого большого числа в столбце.
    # Поскольку индексы совпадают с цифрами (0-9), это и есть предсказанная цифра.
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    # Сравниваем массивы предсказаний и правильных ответов (получаем значения True/False).
    # Суммируем совпадения и делим на общее число картинок.
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    # 1. Задаем стартовые случайные веса и смещения
    W1, b1, W2, b2 = init_params()
    
    # 2. Запускаем цикл обучения
    for i in range(iterations):
        # Шаг 1: Прямое распространение (сеть делает предсказание)
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        
        # Шаг 2: Обратное распространение (сеть считает ошибки и градиенты)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        
        # Шаг 3: Обновление параметров (сеть корректирует веса)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        
        # Каждые 10 итераций выводим прогресс на экран
        if i % 10 == 0:
            print("Итерация: ", i)
            predictions = get_predictions(A2)
            print("Точность на обучающей выборке: ", get_accuracy(predictions, Y))
            
    # Возвращаем натренированные веса! Теперь они содержат знания сети.
    return W1, b1, W2, b2

def save_weights(W1, b1, W2, b2, filename="weights.json"):
    import json
    # Превращаем numpy матрицы в обычные списки Python, 
    # потому что JSON не умеет работать с сырыми массивами numpy
    data = {
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist()
    }
    
    # Открываем файл на запись ('w' - write) и сохраняем туда наши данные
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    print(f"Веса успешно сохранены в файл: {filename}")

def make_predictions(X, W1, b1, W2, b2):
    # Запускаем только прямое распространение (без обучения) на новых данных X
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    
    # Превращаем вероятности в конкретные цифры-ответы
    predictions = get_predictions(A2)
    
    # Возвращаем массив с предсказанными цифрами
    return predictions


if __name__ == "__main__":
    # Указываем пути к бинарным файлам MNIST
    train_images_path = 'train-images-idx3-ubyte'
    train_labels_path = 'train-labels-idx1-ubyte'
    test_images_path = 't10k-images-idx3-ubyte'
    test_labels_path = 't10k-labels-idx1-ubyte'
    
    print("Загрузка данных...")
    X_train, Y_train, X_test, Y_test = load_mnist_idx(
        train_images_path, train_labels_path, test_images_path, test_labels_path
    )
    
    print("Данные загружены. Начинаем обучение...")
    # Запускаем обучение на 500 итераций
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 10000)
    print("Обучение завершено!")
    
    print("---")
    print("Начинаем тестирование на новых данных (X_test)...")
    
    # Делаем предсказания на тестовых данных, которые сеть еще не видела
    test_predictions = make_predictions(X_test, W1, b1, W2, b2)
    
    # Считаем и выводим итоговую точность
    test_acc = get_accuracy(test_predictions, Y_test)
    print(f"Итоговая точность на тестовой выборке: {test_acc * 100:.2f}%")
    
    print("---")
    # Сохраняем натренированные веса для нашего будущего веб-интерфейса
    save_weights(W1, b1, W2, b2, "weights.json")