#version 450

// Оптимизированный размер рабочей группы для лучшей производительности
layout(local_size_x = 32, local_size_y = 32) in;

// Входные буферы с оптимизированным выравниванием
layout(std430, binding = 0) buffer CurrentBuffer {
    float current[];
};

layout(std430, binding = 1) buffer PreviousBuffer {
    float previous[];
};

layout(std430, binding = 2) buffer NextBuffer {
    float next[];
};

layout(std430, binding = 3) buffer ExcitationBuffer {
    float excitation[];
};

// Uniform переменные с оптимизированным выравниванием
layout(std140, binding = 0) uniform FDMUniforms {
    int gridSize;
    float dt;
    float dx;
    float dy;
    float K_coeff;
    float F_coeff;
    float damping;
    float excitationAmplitude;
    float frequency;
    float simulationTime;
    int excitationMode;
    int mParameter;
    float plateRadius;
    float plateWidth;
    float plateHeight;
};

// Оптимизированная функция для получения индекса
int getIndex(int i, int j) {
    return i * gridSize + j;
}

// Оптимизированная функция для безопасного получения значения
float getValue(float[] array, int i, int j) {
    // Используем битовые операции для быстрой проверки границ
    if ((i | j) < 0 || i >= gridSize || j >= gridSize) {
        return 0.0;
    }
    return array[getIndex(i, j)];
}

// Оптимизированная функция Бесселя с предвычисленными константами
float besselJ(int n, float x) {
    float absX = abs(x);
    
    if (absX < 1e-6) {
        return (n == 0) ? 1.0 : 0.0;
    }
    
    // Используем более точные приближения
    if (absX < 3.75) {
        float y = absX * absX * 0.07111111111111111; // 1/14.0625
        float y2 = y * y;
        float y3 = y2 * y;
        float y4 = y3 * y;
        
        float p1 = 1.0 + 0.183105e-2 * y - 0.351639e-4 * y2 + 0.245752e-5 * y3 - 0.240337e-6 * y4;
        float p2 = 0.0468749 + 0.0333190 * y + 0.350834e-3 * y2 - 0.230105e-3 * y3 + 0.367833e-4 * y4;
        
        float result = p1 / p2;
        return (x < 0.0 && (n % 2 != 0)) ? -result : result;
    }
    
    // Оптимизированное приближение для больших x
    float z = 3.75 / absX;
    float z2 = z * z;
    float z3 = z2 * z;
    float z4 = z3 * z;
    float z5 = z4 * z;
    float z6 = z5 * z;
    float z7 = z6 * z;
    float z8 = z7 * z;
    
    float p1 = 0.39894228 + 0.01328592 * z + 0.00225319 * z2 - 0.00157565 * z3 + 
               0.00916281 * z4 - 0.02057706 * z5 + 0.02635537 * z6 - 0.01647633 * z7 + 0.00392377 * z8;
    
    float result = p1 * exp(absX) / sqrt(absX);
    return (x < 0.0 && (n % 2 != 0)) ? -result : result;
}

// Оптимизированная функция для вычисления корня функции Бесселя
float getBesselZero(int m, int n) {
    // Расширенная таблица корней для лучшей точности
    float roots[8][8] = float[8][8](
        float[8](2.4048, 3.8317, 5.1356, 6.3802, 7.5883, 8.7715, 9.9361, 11.0864),
        float[8](3.8317, 5.1356, 6.3802, 7.5883, 8.7715, 9.9361, 11.0864, 12.2251),
        float[8](5.1356, 6.3802, 7.5883, 8.7715, 9.9361, 11.0864, 12.2251, 13.3513),
        float[8](6.3802, 7.5883, 8.7715, 9.9361, 11.0864, 12.2251, 13.3513, 14.4657),
        float[8](7.5883, 8.7715, 9.9361, 11.0864, 12.2251, 13.3513, 14.4657, 15.5699),
        float[8](8.7715, 9.9361, 11.0864, 12.2251, 13.3513, 14.4657, 15.5699, 16.6658),
        float[8](9.9361, 11.0864, 12.2251, 13.3513, 14.4657, 15.5699, 16.6658, 17.7544),
        float[8](11.0864, 12.2251, 13.3513, 14.4657, 15.5699, 16.6658, 17.7544, 18.8363)
    );
    
    if (m >= 0 && m < 8 && n >= 1 && n <= 8) {
        return roots[m][n-1];
    }
    
    // Улучшенное приближение для больших значений
    return 3.14159265359 * (n + 0.5 * m - 0.25);
}

// Оптимизированная функция для вычисления возбуждения
float calculateExcitation(int i, int j) {
    if (excitationMode == 0) {
        // Модальное возбуждение
        float x = (float(j) / float(gridSize - 1) - 0.5) * plateWidth;
        float y = (float(i) / float(gridSize - 1) - 0.5) * plateHeight;
        float r = sqrt(x * x + y * y);
        
        if (r > plateRadius) return 0.0;
        
        float normalizedR = r / plateRadius;
        float besselZero = getBesselZero(mParameter, 1);
        float besselValue = besselJ(mParameter, besselZero * normalizedR);
        
        return excitationAmplitude * besselValue * sin(2.0 * 3.14159265359 * frequency * simulationTime);
    } else {
        // Точечное возбуждение
        float centerX = 0.0;
        float centerY = 0.0;
        float x = (float(j) / float(gridSize - 1) - 0.5) * plateWidth;
        float y = (float(i) / float(gridSize - 1) - 0.5) * plateHeight;
        
        float distance = sqrt((x - centerX) * (x - centerX) + (y - centerY) * (y - centerY));
        float excitationRadius = min(plateWidth, plateHeight) * 0.05;
        
        if (distance < excitationRadius) {
            return excitationAmplitude * sin(2.0 * 3.14159265359 * frequency * simulationTime);
        }
        return 0.0;
    }
}

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int j = int(gl_GlobalInvocationID.y);
    
    // Проверка границ с оптимизацией
    if (i >= gridSize || j >= gridSize) {
        return;
    }
    
    int index = getIndex(i, j);
    
    // Получение текущих значений с оптимизированным доступом
    float currentVal = current[index];
    float prevVal = previous[index];
    
    // Вычисление бигармонического оператора с оптимизацией
    float laplacian = 0.0;
    
    // Центральная точка
    float center = currentVal;
    
    // Соседние точки с проверкой границ
    float left = getValue(current, i, j - 1);
    float right = getValue(current, i, j + 1);
    float up = getValue(current, i - 1, j);
    float down = getValue(current, i + 1, j);
    
    // Диагональные точки для лучшей точности
    float upLeft = getValue(current, i - 1, j - 1);
    float upRight = getValue(current, i - 1, j + 1);
    float downLeft = getValue(current, i + 1, j - 1);
    float downRight = getValue(current, i + 1, j + 1);
    
    // Первый лапласиан
    float laplacian1 = (left + right - 2.0 * center) / (dx * dx) + 
                       (up + down - 2.0 * center) / (dy * dy);
    
    // Второй лапласиан (бигармонический оператор)
    float laplacian2 = (getValue(current, i, j - 2) + getValue(current, i, j + 2) - 2.0 * laplacian1) / (dx * dx) +
                       (getValue(current, i - 2, j) + getValue(current, i + 2, j) - 2.0 * laplacian1) / (dy * dy);
    
    // Вычисление возбуждения
    float excitation = calculateExcitation(i, j);
    
    // Обновление с использованием метода конечных разностей
    float acceleration = K_coeff * laplacian2 + F_coeff * excitation;
    float newVal = 2.0 * currentVal - prevVal + dt * dt * acceleration;
    
    // Применение демпфирования
    newVal *= damping;
    
    // Запись результата
    next[index] = newVal;
}
