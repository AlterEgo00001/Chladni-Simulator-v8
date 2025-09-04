#version 450

layout(local_size_x = 256) in;

// Выходные буферы
layout(std430, binding = 0) buffer ParticlePositions {
    vec4 positions[];
};

layout(std430, binding = 1) buffer ParticleVelocities {
    vec4 velocities[];
};

layout(std430, binding = 2) buffer ParticleStates {
    int states[];
};

// Uniform переменные
layout(std140, binding = 0) uniform InitParticleUniforms {
    int particleCount;
    float plateRadius;
    float randomSeed;
};

// Простая функция генерации случайных чисел
float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

void main() {
    int particleIndex = int(gl_GlobalInvocationID.x);
    
    if (particleIndex >= particleCount) {
        return;
    }
    
    // Генерация случайной позиции внутри круга
    vec2 randomVec = vec2(random(vec2(float(particleIndex), 0.0)), 
                          random(vec2(float(particleIndex), 1.0)));
    
    float angle = 2.0 * 3.14159 * randomVec.x;
    float radius = plateRadius * sqrt(randomVec.y);
    
    vec3 position = vec3(radius * cos(angle), radius * sin(angle), 0.0);
    vec3 velocity = vec3(0.0, 0.0, 0.0);
    
    positions[particleIndex] = vec4(position, 1.0);
    velocities[particleIndex] = vec4(velocity, 1.0);
    states[particleIndex] = 0; // visible
}
