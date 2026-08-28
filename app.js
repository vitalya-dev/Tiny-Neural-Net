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
    // Очищаем большой основной холст
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Находим наш новый маленький холст и тоже заливаем его черным
    const networkCanvas = document.getElementById('networkInputCanvas');
    if (networkCanvas) {
        const networkCtx = networkCanvas.getContext('2d');
        networkCtx.fillStyle = 'black';
        networkCtx.fillRect(0, 0, 28, 28);
    }
    
    // Сбрасываем текст с предсказанием
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
    const ctx = canvas.getContext('2d');
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;

    // 1. Ищем границы рисунка
    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
    let isEmpty = true;

    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const red = data[(y * canvas.width + x) * 4]; 
            if (red > 0) { 
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                isEmpty = false;
            }
        }
    }

    // Берем наш новый маленький холст прямо со страницы
    const networkCanvas = document.getElementById('networkInputCanvas');
    const networkCtx = networkCanvas.getContext('2d');

    // Если ничего не нарисовано, очищаем маленький холст и возвращаем нули
    if (isEmpty) {
        networkCtx.fillStyle = 'black';
        networkCtx.fillRect(0, 0, 28, 28);
        return new Array(784).fill(0);
    }

    const digitWidth = maxX - minX + 1;
    const digitHeight = maxY - minY + 1;

    // 2. Вырезаем цифру на временный холст
    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = digitWidth;
    croppedCanvas.height = digitHeight;
    const croppedCtx = croppedCanvas.getContext('2d');
    croppedCtx.drawImage(canvas, minX, minY, digitWidth, digitHeight, 0, 0, digitWidth, digitHeight);

    // 3. Вычисляем масштаб (вписываем в 20x20)
    const scale = 20 / Math.max(digitWidth, digitHeight);
    const scaledWidth = digitWidth * scale;
    const scaledHeight = digitHeight * scale;

    // 4. Обязательно заливаем фон видимого маленького холста черным
    networkCtx.fillStyle = 'black';
    networkCtx.fillRect(0, 0, 28, 28);

    // 5. Размещаем сжатую цифру ровно по центру на видимом маленьком холсте!
    const dx = (28 - scaledWidth) / 2;
    const dy = (28 - scaledHeight) / 2;
    networkCtx.drawImage(croppedCanvas, 0, 0, digitWidth, digitHeight, dx, dy, scaledWidth, scaledHeight);

    // 6. Получаем финальные пиксели прямо с видимого холста
    const finalData = networkCtx.getImageData(0, 0, 28, 28).data;
    const pixels = [];
    
    for (let i = 0; i < finalData.length; i += 4) {
        pixels.push(finalData[i] / 255.0);
    }

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


