#version 450

layout(local_size_x = 16, local_size_y = 16) in;

// Входные буферы
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

// Uniform переменные
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

// Функция для получения индекса в 2D массиве
int getIndex(int i, int j) {
    return i * gridSize + j;
}

// Функция для безопасного получения значения из массива
float getValue(float[] array, int i, int j) {
    if (i < 0 || i >= gridSize || j < 0 || j >= gridSize) {
        return 0.0;
    }
    return array[getIndex(i, j)];
}

// Функция Бесселя первого рода (упрощенная версия)
float besselJ(int n, float x) {
    if (x < 0.0) {
        return (n % 2 == 0) ? besselJ(n, -x) : -besselJ(n, -x);
    }
    
    if (x == 0.0) {
        return (n == 0) ? 1.0 : 0.0;
    }
    
    // Приближение для малых x
    if (x < 3.75) {
        float y = x * x / 14.0625;
        float p1 = 1.0 + 0.183105e-2 * y - 0.351639e-4 * y * y + 0.245752e-5 * y * y * y - 0.240337e-6 * y * y * y * y;
        float p2 = 0.0468749 + 0.0333190 * y + 0.350834e-3 * y * y - 0.230105e-3 * y * y * y + 0.367833e-4 * y * y * y * y;
        return p1 / p2;
    }
    
    // Приближение для больших x
    float z = 3.75 / x;
    float p1 = 0.39894228 + 0.01328592 * z + 0.00225319 * z * z - 0.00157565 * z * z * z + 0.00916281 * z * z * z * z - 0.02057706 * z * z * z * z * z + 0.02635537 * z * z * z * z * z * z - 0.01647633 * z * z * z * z * z * z * z + 0.00392377 * z * z * z * z * z * z * z * z;
    return p1 * exp(x) / sqrt(x);
}

// Функция для вычисления корня функции Бесселя
float getBesselZero(int m, int n) {
    // Таблица корней функций Бесселя (первые несколько значений)
    float roots[5][5] = float[5][5](
        float[5](2.4048, 3.8317, 5.1356, 6.3802, 7.5883),
        float[5](3.8317, 5.1356, 6.3802, 7.5883, 8.7715),
        float[5](5.1356, 6.3802, 7.5883, 8.7715, 9.9361),
        float[5](6.3802, 7.5883, 8.7715, 9.9361, 11.0864),
        float[5](7.5883, 8.7715, 9.9361, 11.0864, 12.2251)
    );
    
    if (m >= 0 && m < 5 && n >= 1 && n <= 5) {
        return roots[m][n-1];
    }
    
    // Приближение для больших значений
    return 3.14159 * (n + 0.5 * m - 0.25);
}

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int j = int(gl_GlobalInvocationID.y);
    
    if (i >= gridSize || j >= gridSize) {
        return;
    }
    
    // Нормализованные координаты
    float normX = (float(j) / float(gridSize - 1)) - 0.5;
    float normY = (float(i) / float(gridSize - 1)) - 0.5;
    
    // Физические координаты
    float physX = normX * plateWidth;
    float physY = normY * plateHeight;
    
    // Проверка границ пластины
    if (physX * physX + physY * physY > plateRadius * plateRadius + dx * dx * 0.25) {
        next[getIndex(i, j)] = 0.0;
        return;
    }
    
    // Текущее значение
    float u_ij = getValue(current, i, j);
    
    // Соседние значения для бигармонического оператора
    float u_ip1j = getValue(current, i + 1, j);
    float u_im1j = getValue(current, i - 1, j);
    float u_ijp1 = getValue(current, i, j + 1);
    float u_ijm1 = getValue(current, i, j - 1);
    float u_ip1jp1 = getValue(current, i + 1, j + 1);
    float u_ip1jm1 = getValue(current, i + 1, j - 1);
    float u_im1jp1 = getValue(current, i - 1, j + 1);
    float u_im1jm1 = getValue(current, i - 1, j - 1);
    float u_ip2j = getValue(current, i + 2, j);
    float u_im2j = getValue(current, i - 2, j);
    float u_ijp2 = getValue(current, i, j + 2);
    float u_ijm2 = getValue(current, i, j - 2);
    
    // Бигармонический оператор
    float inv_dx4 = 1.0 / (dx * dx * dx * dx);
    float biharmonic = (20.0 * u_ij - 8.0 * (u_ip1j + u_im1j + u_ijp1 + u_ijm1) + 
                        2.0 * (u_ip1jp1 + u_ip1jm1 + u_im1jp1 + u_im1jm1) + 
                        (u_ip2j + u_im2j + u_ijp2 + u_ijm2)) * inv_dx4;
    
    // Возбуждающая сила
    float excForce = 0.0;
    float timeSine = sin(2.0 * 3.14159 * frequency * simulationTime);
    
    if (excitationMode == 0) {
        // Модальное возбуждение
        float theta = atan(physY, physX);
        float besselZero = getBesselZero(mParameter, 1);
        float k_mn = besselZero / plateRadius;
        float r_phys = sqrt(physX * physX + physY * physY);
        
        if (r_phys <= plateRadius + dx * 0.15) {
            float besselValue = besselJ(mParameter, k_mn * r_phys);
            excForce = excitationAmplitude * timeSine * besselValue * cos(float(mParameter) * theta);
        }
    } else {
        // Точечное возбуждение
        int centerI = gridSize / 2;
        int centerJ = gridSize / 2;
        float distSq = float((i - centerI) * (i - centerI) + (j - centerJ) * (j - centerJ));
        float excRadSq = max(1.0, float(gridSize * gridSize) * 0.0016);
        
        if (distSq <= excRadSq) {
            excForce = excitationAmplitude * timeSine * exp(-distSq / (excRadSq * 0.5 + 1e-9));
        }
    }
    
    // Обновление значения
    float u_next = (2.0 * u_ij - getValue(previous, i, j)) - K_coeff * biharmonic + F_coeff * excForce;
    next[getIndex(i, j)] = isfinite(u_next) ? u_next * (1.0 - damping) : 0.0;
}
