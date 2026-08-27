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


function initApp() {
    const { canvas, ctx } = setupCanvas();
    let isDrawing = false; // Переменная, которая помнит, рисуем мы сейчас или нет
    
    // Очищаем холст при старте, чтобы он был гарантированно черным
    clearCanvas(canvas, ctx);

    // Добавляем "слушателей" событий мыши
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = startDrawing(e, ctx);
    });

    canvas.addEventListener('mousemove', (e) => {
        draw(e, ctx, isDrawing);
    });

    canvas.addEventListener('mouseup', () => {
        isDrawing = stopDrawing(ctx);
    });

    canvas.addEventListener('mouseout', () => {
        isDrawing = stopDrawing(ctx);
    });

    // Настраиваем кнопку очистки
    const clearBtn = document.getElementById('clearBtn');
    clearBtn.addEventListener('click', () => {
        clearCanvas(canvas, ctx);
    });
}

// Запускаем наше приложение, как только страница загрузится
window.onload = initApp;


