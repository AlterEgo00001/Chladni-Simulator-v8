#version 450

layout(local_size_x = 16, local_size_y = 16) in;

// Выходные буферы
layout(std430, binding = 0) buffer CurrentBuffer {
    float current[];
};

layout(std430, binding = 1) buffer PreviousBuffer {
    float previous[];
};

layout(std430, binding = 2) buffer ExcitationBuffer {
    float excitation[];
};

// Uniform переменные
layout(std140, binding = 0) uniform InitUniforms {
    int gridSize;
    float plateRadius;
    float plateWidth;
    float plateHeight;
    float dx;
    float dy;
    int mParameter;
    int nParameter;
    int drivingMechanism;
    float amplitudeScale;
};

// Функция для получения индекса в 2D массиве
int getIndex(int i, int j) {
    return i * gridSize + j;
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
    
    int index = getIndex(i, j);
    
    // Инициализация нулевыми значениями
    current[index] = 0.0;
    previous[index] = 0.0;
    excitation[index] = 0.0;
    
    if (drivingMechanism == 0) { // modal
        // Нормализованные координаты
        float normX = (float(j) / float(gridSize - 1)) - 0.5;
        float normY = (float(i) / float(gridSize - 1)) - 0.5;
        
        // Физические координаты
        float physX = normX * plateWidth;
        float physY = normY * plateHeight;
        
        float r_phys = sqrt(physX * physX + physY * physY);
        
        if (r_phys <= plateRadius + dx * 0.15) {
            float b_zero = getBesselZero(mParameter, nParameter);
            if (b_zero > 0.0) {
                float k_mn = b_zero / plateRadius;
                float theta_phys = atan(physY, physX);
                float disp = besselJ(mParameter, k_mn * r_phys) * cos(float(mParameter) * theta_phys);
                
                current[index] = disp * amplitudeScale;
                previous[index] = disp * amplitudeScale;
                excitation[index] = besselJ(mParameter, k_mn * r_phys);
            }
        }
    }
}
