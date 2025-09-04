#version 450

// Оптимизированный размер рабочей группы для частиц
layout(local_size_x = 512) in;

// Входные буферы с оптимизированным выравниванием
layout(std430, binding = 0) buffer DisplacementBuffer {
    float displacement[];
};

layout(std430, binding = 1) buffer ParticlePositions {
    vec4 positions[];
};

layout(std430, binding = 2) buffer ParticleVelocities {
    vec4 velocities[];
};

layout(std430, binding = 3) buffer ParticleStates {
    int states[];
};

// Выходные буферы
layout(std430, binding = 4) buffer OutputPositions {
    vec4 outputPositions[];
};

layout(std430, binding = 5) buffer OutputVelocities {
    vec4 outputVelocities[];
};

layout(std430, binding = 6) buffer OutputStates {
    int outputStates[];
};

// Uniform переменные с оптимизированным выравниванием
layout(std140, binding = 0) uniform ParticleUniforms {
    int gridSize;
    int particleCount;
    float deltaTime;
    float plateRadius;
    float plateWidth;
    float plateHeight;
    float dx;
    float dy;
    float forceMultiplier;
    float damping;
    float repulsionRadius;
    float repulsionStrength;
    float maxSpeed;
    float restitution;
    float visualDeformScale;
    float maxVisualAmplitude;
    float plateRotationAngle;
    int enableRepulsion;
    int maxRepulsionNeighbors;
    float stuckVelocityThreshold;
    float stuckDispThreshold;
    int enableStuckCulling;
    vec3 hiddenPosition;
    float actualAppliedFrequency;
};

// Оптимизированная функция для получения индекса
int getIndex(int i, int j) {
    return i * gridSize + j;
}

// Оптимизированная функция для безопасного получения значения
float getDisplacement(int i, int j) {
    if ((i | j) < 0 || i >= gridSize || j >= gridSize) {
        return 0.0;
    }
    return displacement[getIndex(i, j)];
}

// Оптимизированная функция для получения смещения из поля FDM
float getDisplacementFromField(vec2 position) {
    float normGX = (position.x / plateWidth) + 0.5;
    float normGY = (position.y / plateHeight) + 0.5;
    float fdmR = normGY * float(gridSize - 1);
    float fdmC = normGX * float(gridSize - 1);
    
    int r = int(fdmR);
    int c = int(fdmC);
    
    // Оптимизированная билинейная интерполяция
    float fracR = fdmR - float(r);
    float fracC = fdmC - float(c);
    
    // Предвычисленные значения для оптимизации
    float d00 = getDisplacement(r, c);
    float d01 = getDisplacement(r, c + 1);
    float d10 = getDisplacement(r + 1, c);
    float d11 = getDisplacement(r + 1, c + 1);
    
    // Оптимизированная интерполяция
    float d0 = mix(d00, d01, fracC);
    float d1 = mix(d10, d11, fracC);
    return mix(d0, d1, fracR);
}

// Оптимизированная функция для получения градиента
vec2 getGradientFromField(vec2 position) {
    float normGX = (position.x / plateWidth) + 0.5;
    float normGY = (position.y / plateHeight) + 0.5;
    float fdmR = normGY * float(gridSize - 1);
    float fdmC = normGX * float(gridSize - 1);
    
    int r = int(round(fdmR));
    int c = int(round(fdmC));
    
    // Оптимизированное вычисление градиента
    float dx = (getDisplacement(r, c + 1) - getDisplacement(r, c - 1)) / (2.0 * dx);
    float dy = (getDisplacement(r + 1, c) - getDisplacement(r - 1, c)) / (2.0 * dy);
    
    return vec2(dx, dy);
}

// Оптимизированная функция для вычисления отталкивания
vec2 calculateRepulsion(vec2 pos, int currentIndex) {
    vec2 repulsionForce = vec2(0.0, 0.0);
    int neighborsChecked = 0;
    
    // Оптимизированный цикл с ранним выходом
    for (int otherIndex = 0; otherIndex < particleCount && neighborsChecked < maxRepulsionNeighbors; otherIndex++) {
        if (otherIndex == currentIndex) continue;
        
        int otherState = states[otherIndex];
        if (otherState == 1) continue; // hidden particle
        
        vec3 otherPos = positions[otherIndex].xyz;
        vec2 delta = pos - otherPos.xy;
        float distSq = dot(delta, delta);
        
        // Оптимизированная проверка расстояния
        if (distSq < repulsionRadius * repulsionRadius && distSq > 1e-9) {
            float dist = sqrt(distSq);
            float repulsionMagnitude = repulsionStrength * pow(repulsionRadius - dist, 2.0) / (dist + 1e-9);
            repulsionForce += repulsionMagnitude * delta / dist;
            neighborsChecked++;
        }
    }
    
    return repulsionForce;
}

void main() {
    int particleIndex = int(gl_GlobalInvocationID.x);
    
    if (particleIndex >= particleCount) {
        return;
    }
    
    vec3 pos = positions[particleIndex].xyz;
    vec3 vel = velocities[particleIndex].xyz;
    int state = states[particleIndex];
    
    // Проверка скрытых частиц
    if (state == 1) {
        outputPositions[particleIndex] = vec4(hiddenPosition, 1.0);
        outputVelocities[particleIndex] = vec4(0.0, 0.0, 0.0, 1.0);
        outputStates[particleIndex] = 1;
        return;
    }
    
    // Получение смещения и градиента из поля FDM
    float disp = getDisplacementFromField(pos.xy);
    vec2 grad = getGradientFromField(pos.xy);
    
    // Вычисление силы от поля с адаптивным множителем
    vec2 force = -2.0 * disp * grad * forceMultiplier;
    
    // Отталкивание между частицами (оптимизированное)
    if (enableRepulsion == 1) {
        force += calculateRepulsion(pos.xy, particleIndex);
    }
    
    // Применение демпфирования с адаптацией для низких частот
    float adaptDamp = damping;
    float forceMult = forceMultiplier;
    
    if (actualAppliedFrequency < 300.0) {
        adaptDamp = min(0.97, damping + 0.03);
        forceMult = forceMultiplier * 1.25;
    }
    
    float effectiveDamping = abs(disp) < 1e-4 ? 0.99 : adaptDamp;
    vel.xy = (vel.xy + force * deltaTime) * effectiveDamping;
    
    // Ограничение скорости с оптимизацией
    float speed = length(vel.xy);
    if (speed > maxSpeed) {
        vel.xy *= maxSpeed / speed;
    }
    
    // Обновление позиции
    pos.xy += vel.xy * deltaTime;
    
    // Оптимизированная проверка границ пластины
    float radiusAfter = length(pos.xy);
    if (radiusAfter > plateRadius) {
        float correctionRatio = plateRadius / radiusAfter;
        pos.xy *= correctionRatio;
        
        vec2 normal = pos.xy / plateRadius;
        float velocityDotNormal = dot(vel.xy, normal);
        
        if (velocityDotNormal > 0.0) {
            vel.xy -= (1.0 + restitution) * velocityDotNormal * normal;
        }
    }
    
    // Скрытие застрявших частиц
    if (enableStuckCulling == 1) {
        if (speed < stuckVelocityThreshold && abs(disp) > stuckDispThreshold) {
            outputPositions[particleIndex] = vec4(hiddenPosition, 1.0);
            outputVelocities[particleIndex] = vec4(0.0, 0.0, 0.0, 1.0);
            outputStates[particleIndex] = 1;
            return;
        }
    }
    
    // Визуальная деформация
    float visualHeight = clamp(disp * visualDeformScale, -maxVisualAmplitude, maxVisualAmplitude);
    
    // Оптимизированное применение вращения пластины
    float cosRot = cos(plateRotationAngle);
    float sinRot = sin(plateRotationAngle);
    float rotX = pos.x * cosRot - pos.y * sinRot;
    float rotZ = pos.x * sinRot + pos.y * cosRot;
    
    outputPositions[particleIndex] = vec4(rotX, visualHeight, rotZ, 1.0);
    outputVelocities[particleIndex] = vec4(vel, 1.0);
    outputStates[particleIndex] = 0;
}
