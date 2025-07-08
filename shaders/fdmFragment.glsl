// File: /shaders/fdmFragment.glsl
// Вычисляет смещение пластины (u_next) на основе предыдущих состояний (u_curr, u_prev).

#define PI 3.141592653589793

// Uniforms из JavaScript
uniform sampler2D uTexture;
uniform sampler2D modalPatternTexture;
uniform vec2 resolution;
uniform float dx;
uniform float dt;
uniform float D_flexural;
uniform float rho_h;
uniform float freq;
uniform float damp;
uniform float excAmp;
uniform float time;
uniform float plateRadius;
uniform int mParam;
uniform int excMode;

// Безопасная выборка из текстуры с учетом границ круглой пластины
// Реализует условие "свободного края" (free-edge boundary condition)
float sample_boundary(vec2 uv, vec2 offset) {
  vec2 sample_uv = uv + offset;
  vec2 sample_phys = (sample_uv - 0.5) * plateRadius * 2.0;
  if (length(sample_phys) > plateRadius) {
    return texture2D(uTexture, uv - offset).r;
  }
  return texture2D(uTexture, sample_uv).r;
}

void main() {
  vec2 uv = gl_FragCoord.xy / resolution;
  vec2 texelSize = 1.0 / resolution;

  float physX = (uv.x - 0.5) * plateRadius * 2.0;
  float physY = (uv.y - 0.5) * plateRadius * 2.0;
  if (length(vec2(physX, physY)) > plateRadius) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
    return;
  }

  vec4 data = texture2D(uTexture, uv);
  float u_curr = data.r;
  float u_prev = data.g;

  // Бигармонический оператор с использованием безопасной выборки
  float inv_dx4 = 1.0 / (dx * dx * dx * dx);
  float u_ip1j = sample_boundary(uv, vec2(0.0, texelSize.y));
  float u_im1j = sample_boundary(uv, vec2(0.0, -texelSize.y));
  float u_ijp1 = sample_boundary(uv, vec2(texelSize.x, 0.0));
  float u_ijm1 = sample_boundary(uv, vec2(-texelSize.x, 0.0));
  float u_ip1jp1 = sample_boundary(uv, vec2(texelSize.x, texelSize.y));
  float u_ip1jm1 = sample_boundary(uv, vec2(texelSize.x, -texelSize.y));
  float u_im1jp1 = sample_boundary(uv, vec2(-texelSize.x, texelSize.y));
  float u_im1jm1 = sample_boundary(uv, vec2(-texelSize.x, -texelSize.y));
  float u_ip2j = sample_boundary(uv, vec2(0.0, 2.0 * texelSize.y));
  float u_im2j = sample_boundary(uv, vec2(0.0, -2.0 * texelSize.y));
  float u_ijp2 = sample_boundary(uv, vec2(2.0 * texelSize.x, 0.0));
  float u_ijm2 = sample_boundary(uv, vec2(-2.0 * texelSize.x, 0.0));

  float biharmonic = (20.0 * u_curr - 8.0 * (u_ip1j + u_im1j + u_ijp1 + u_ijm1) +
                     2.0 * (u_ip1jp1 + u_ip1jm1 + u_im1jp1 + u_im1jm1) +
                     (u_ip2j + u_im2j + u_ijp2 + u_ijm2)) * inv_dx4;

  // Возбуждающая сила
  float excForce = 0.0;
  float timeSine = sin(2.0 * PI * freq * time);
  if (excMode == 0) {
    float theta = atan(physY, physX);
    float modalPattern = texture2D(modalPatternTexture, uv).r;
    excForce = excAmp * timeSine * modalPattern * cos(float(mParam) * theta);
  } else {
    vec2 centerUV = vec2(0.5, 0.5);
    float distSq = dot(uv - centerUV, uv - centerUV);
    if (distSq <= 0.0025) { // Возбуждение в небольшом радиусе от центра
      excForce = excAmp * timeSine * exp(-distSq / 0.001);
    }
  }

  // Финальная формула
  float K_coeff = (dt * dt * D_flexural) / rho_h;
  float F_coeff = (dt * dt) / rho_h;
  float u_next = (2.0 * u_curr - u_prev) - K_coeff * biharmonic + F_coeff * excForce;
  u_next *= (1.0 - damp);

  gl_FragColor = vec4(u_next, u_curr, 0.0, 0.0);
}