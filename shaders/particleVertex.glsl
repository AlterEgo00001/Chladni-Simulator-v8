// File: /shaders/particleVertex.glsl
// Позиционирует каждую частицу в 3D пространстве для рендеринга.

attribute float instanceId;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform sampler2D particleTexture;
uniform sampler2D displacementTexture;
uniform vec2 particleTexResolution;
uniform float plateWidth;
uniform float visScale;
uniform float maxVisAmp;
uniform float rotationAngle;
uniform float u_activeParticleCount;

varying vec3 v_worldPosition;
varying vec3 v_normal;
varying vec2 v_particleUV;

void main() {
  // Отсекаем "неактивные" частицы для динамической плотности
  if (instanceId >= u_activeParticleCount) {
    gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  v_particleUV = vec2(
    mod(instanceId, particleTexResolution.x) + 0.5,
    floor(instanceId / particleTexResolution.x) + 0.5
  ) / particleTexResolution;
  
  vec4 data = texture2D(particleTexture, v_particleUV);
  vec2 pos2D = data.rg;

  // Если частица скрыта, не рендерим ее
  if (pos2D.x > 900.0) {
    gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  // Высота частицы
  vec2 normPos = pos2D / plateWidth + 0.5;
  float disp = texture2D(displacementTexture, normPos).r;
  float visHeight = clamp(disp * visScale, -maxVisAmp, maxVisAmp);

  // Вращение
  float cosA = cos(rotationAngle);
  float sinA = sin(rotationAngle);
  float rotX = pos2D.x * cosA - pos2D.y * sinA;
  float rotZ = pos2D.x * sinA + pos2D.y * cosA;
  
  vec3 finalOffset = vec3(rotX, visHeight, rotZ);
  
  // Данные для освещения
  v_worldPosition = finalOffset + position;
  v_normal = normalize(position);

  // Финальная позиция
  vec4 mvPosition = modelViewMatrix * vec4(v_worldPosition, 1.0);
  gl_Position = projectionMatrix * mvPosition;
}