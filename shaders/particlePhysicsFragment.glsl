// File: /shaders/particlePhysicsFragment.glsl
// Обновляет позицию и скорость каждой частицы на основе поля смещений и сил отталкивания.

uniform sampler2D particleTexture;
uniform sampler2D displacementTexture;
uniform vec2 resolution;
uniform float plateRadius;
uniform float plateWidth;
uniform float dx;
uniform float dy;
uniform float forceMult;
uniform float damping;
uniform float restitution;
uniform float maxSpeed;
uniform float deltaTime;
uniform float repulsionRadius;
uniform float repulsionStrength;
uniform float stuckThreshold;

void main() {
  vec2 uv = gl_FragCoord.xy / resolution;
  vec4 data = texture2D(particleTexture, uv);
  vec2 pos = data.rg;
  vec2 vel = data.ba;

  // Если частица скрыта, не обрабатываем ее
  if (pos.x > 900.0) {
    gl_FragColor = vec4(pos, vel);
    return;
  }

  // Сила от градиента
  vec2 normPos = pos / plateWidth + 0.5;
  float disp = texture2D(displacementTexture, normPos).r;
  vec2 texelSizeDisp = 1.0 / vec2(textureSize(displacementTexture, 0));
  float gradX = (texture2D(displacementTexture, normPos + vec2(texelSizeDisp.x, 0.0)).r - texture2D(displacementTexture, normPos - vec2(texelSizeDisp.x, 0.0)).r) / (2.0 * dx);
  float gradY = (texture2D(displacementTexture, normPos + vec2(0.0, texelSizeDisp.y)).r - texture2D(displacementTexture, normPos - vec2(0.0, texelSizeDisp.y)).r) / (2.0 * dy);
  vec2 force = -2.0 * disp * vec2(gradX, gradY) * forceMult;

  // Сила отталкивания
  if (repulsionStrength > 0.0) {
    vec2 texelSizeParticle = 1.0 / resolution;
    vec2 repulsionForce = vec2(0.0);
    const int checks = 4;
    for (int i = 1; i <= checks; i++) {
      float angle = float(i) / float(checks) * 6.283185;
      vec2 neighborUV = uv + vec2(cos(angle), sin(angle)) * texelSizeParticle;
      vec2 toNeighbor = texture2D(particleTexture, neighborUV).rg - pos;
      float distSq = dot(toNeighbor, toNeighbor);
      if (distSq < repulsionRadius * repulsionRadius && distSq > 0.00001) {
        float dist = sqrt(distSq);
        repulsionForce -= ((repulsionRadius - dist) / repulsionRadius) * normalize(toNeighbor) * repulsionStrength;
      }
    }
    force += repulsionForce;
  }

  // Обновление скорости и позиции
  vel = (vel + force * deltaTime) * damping;
  if (length(vel) > maxSpeed) vel = normalize(vel) * maxSpeed;
  pos += vel * deltaTime;

  // Столкновения
  if (length(pos) > plateRadius) {
    pos = normalize(pos) * plateRadius;
    vec2 normB = pos / plateRadius;
    vel -= (1.0 + restitution) * dot(vel, normB) * normB;
  }

  // Скрытие застрявших частиц
  if (length(vel) < stuckThreshold && abs(disp) > 0.05) {
    pos = vec2(1001.0, 1001.0);
    vel = vec2(0.0, 0.0);
  }

  gl_FragColor = vec4(pos, vel);
}