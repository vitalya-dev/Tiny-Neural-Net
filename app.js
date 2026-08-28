function setupCanvas() {
    const canvas = document.getElementById('paintCanvas');
    const ctx = canvas.getContext('2d');
    
    // Настраиваем стиль кисти
    // Толщина 15 пикселей будет отлично смотреться при сжатии до 28x28
    ctx.lineWidth = 15; 
    ctx.lineCap = 'round'; // Круглые края линий для плавности
    ctx.strokeStyle = 'white'; // Белая краска
    
    return { canvas, ctx };
}

function startDrawing(e, ctx) {
    // Говорим браузеру, что начинаем новый путь (линию)
    ctx.beginPath();
    // Ставим кисть в те координаты, где сейчас находится мышка
    ctx.moveTo(e.offsetX, e.offsetY);
    // Возвращаем true, что значит "режим рисования включен"
    return true; 
}

function draw(e, ctx, isDrawing) {
    // Если мышка не нажата, просто выходим из функции
    if (!isDrawing) return;
    
    // Проводим невидимую линию до новых координат
    ctx.lineTo(e.offsetX, e.offsetY);
    // Обводим эту линию цветом (делаем видимой)
    ctx.stroke();
}

function stopDrawing(ctx) {
    // Завершаем текущий путь
    ctx.closePath();
    // Возвращаем false, что значит "режим рисования выключен"
    return false; 
}

function clearCanvas(canvas, ctx) {
    // Заливаем весь холст черным цветом (это важно для нейросети)
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Сбрасываем текст с предсказанием к начальному виду
    const predictionText = document.getElementById('predictionText');
    if (predictionText) {
        predictionText.innerText = "Предсказание: ?";
    }
}


async function initApp() {
    const { canvas, ctx } = setupCanvas();
    let isDrawing = false;
    
    // Переменная для хранения наших натренированных весов
    let weights = null;

    clearCanvas(canvas, ctx);

    // Пытаемся загрузить веса из файла weights.json
    try {
        const response = await fetch('weights.json');
        weights = await response.json();
        console.log("Веса успешно загружены!");
    } catch (error) {
        console.error("Ошибка загрузки весов:", error);
        document.getElementById('predictionText').innerText = "Ошибка загрузки весов";
    }

    // Функция, которая объединяет получение пикселей и работу нейросети
    function makePrediction() {
        if (!weights) return; // Если веса еще не скачались, ничего не делаем
        
        const pixels = getPixels(canvas);
        const probabilities = forwardPropJS(pixels, weights.W1, weights.b1, weights.W2, weights.b2);
        const predictedDigit = argmax(probabilities);
        
        // Выводим результат на экран
        document.getElementById('predictionText').innerText = `Предсказание: ${predictedDigit}`;
    }

    canvas.addEventListener('mousedown', (e) => {
        isDrawing = startDrawing(e, ctx);
    });

    canvas.addEventListener('mousemove', (e) => {
        draw(e, ctx, isDrawing);
    });

    canvas.addEventListener('mouseup', () => {
        isDrawing = stopDrawing(ctx);
        // Как только закончили рисовать линию - делаем предсказание!
        makePrediction(); 
    });

    canvas.addEventListener('mouseout', () => {
        isDrawing = stopDrawing(ctx);
    });

    const clearBtn = document.getElementById('clearBtn');
    clearBtn.addEventListener('click', () => {
        clearCanvas(canvas, ctx);
    });
}


function getPixels(canvas) {
    // 1. Создаем временный холст нужного размера (28x28)
    const scaledCanvas = document.createElement('canvas');
    scaledCanvas.width = 28;
    scaledCanvas.height = 28;
    const scaledCtx = scaledCanvas.getContext('2d');
    
    // 2. Рисуем большую картинку на маленьком холсте (сжатие)
    scaledCtx.drawImage(canvas, 0, 0, 28, 28);
    
    // 3. Получаем данные пикселей
    // getImageData возвращает массив, где каждый пиксель описан 4 числами: 
    // Красный (R), Зеленый (G), Синий (B) и Прозрачность (Alpha).
    const imageData = scaledCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    
    // 4. Массив для нашей нейросети (784 элемента)
    const pixels = [];
    
    // 5. Проходим по массиву с шагом 4, чтобы брать только первый цвет (Красный)
    // Так как мы рисуем белым по черному, R, G и B у нас одинаковые (255 или 0)
    for (let i = 0; i < data.length; i += 4) {
        // Берем значение цвета (от 0 до 255) и делим на 255, 
        // чтобы получить нормализованное значение (от 0.0 до 1.0)
        const normalizedPixel = data[i] / 255.0;
        pixels.push(normalizedPixel);
    }
    
    // Возвращаем готовый массив из 784 чисел
    return pixels;
}

function relu(vector) {
    // Проходим по каждому числу в массиве. Если оно меньше 0, делаем его 0.
    return vector.map(val => Math.max(0, val));
}

function softmax(vector) {
    // Находим максимальное значение (полезно для стабильности вычислений)
    const maxVal = Math.max(...vector); 
    // Возводим в экспоненту
    const exps = vector.map(val => Math.exp(val - maxVal));
    // Считаем сумму всех экспонент
    const sumExps = exps.reduce((a, b) => a + b, 0);
    // Делим каждое значение на сумму, получая вероятности от 0 до 1
    return exps.map(val => val / sumExps);
}

function matrixVectorMultiplyAndAdd(matrix, vector, bias) {
    const result = [];
    // Проходим по каждой строке матрицы (весов)
    for (let i = 0; i < matrix.length; i++) {
        let sum = 0;
        // Умножаем на пиксели картинки
        for (let j = 0; j < vector.length; j++) {
            sum += matrix[i][j] * vector[j];
        }
        // Добавляем смещение (bias). В JSON это массив массивов (n x 1), берем первый элемент
        result.push(sum + bias[i][0]);
    }
    return result;
}

function forwardPropJS(X, W1, b1, W2, b2) {
    // Повторяем логику Python: сначала первый слой и ReLU
    const Z1 = matrixVectorMultiplyAndAdd(W1, X, b1);
    const A1 = relu(Z1);
    
    // Затем второй слой и Softmax
    const Z2 = matrixVectorMultiplyAndAdd(W2, A1, b2);
    const A2 = softmax(Z2);
    
    // Возвращаем массив вероятностей для каждой цифры
    return A2;
}

function argmax(vector) {
    // Ищем индекс самого большого числа в массиве (это и есть наша угаданная цифра)
    let maxIndex = 0;
    let maxValue = vector[0];
    for (let i = 1; i < vector.length; i++) {
        if (vector[i] > maxValue) {
            maxValue = vector[i];
            maxIndex = i;
        }
    }
    return maxIndex;
}

// Запускаем наше приложение, как только страница загрузится
window.onload = initApp;


