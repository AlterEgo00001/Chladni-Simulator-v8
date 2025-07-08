// File: /shaders/particleFragment.glsl
// Рассчитывает цвет частицы на основе режима (по скорости или по частоте) и освещения.

varying vec3 v_worldPosition;
varying vec3 v_normal;
varying vec2 v_particleUV;

uniform sampler2D particleTexture;
uniform vec3 lightPos;
uniform vec3 cameraPos;
uniform float u_colorMode;
uniform vec3 u_globalColor;
uniform float maxSpeedForColor;
uniform vec3 coldColor;
uniform vec3 hotColor;

void main() {
  vec3 baseColor;

  if (u_colorMode > 0.5) {
    // Цвет по скорости
    vec2 vel = texture2D(particleTexture, v_particleUV).ba;
    float speed = length(vel);
    float speedFactor = clamp(speed / maxSpeedForColor, 0.0, 1.0);
    baseColor = mix(coldColor, hotColor, speedFactor);
  } else {
    // Глобальный цвет (по частоте)
    baseColor = u_globalColor;
  }

  // Освещение по Фонгу
  vec3 normal = normalize(v_normal);
  vec3 lightDir = normalize(lightPos - v_worldPosition);
  vec3 viewDir = normalize(cameraPos - v_worldPosition);
  vec3 reflectDir = reflect(-lightDir, normal);
  
  float ambientStrength = 0.3;
  vec3 ambient = ambientStrength * baseColor;
  
  float diff = max(dot(normal, lightDir), 0.0);
  vec3 diffuse = diff * baseColor;
  
  float specularStrength = 0.8;
  float shininess = 32.0;
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
  vec3 specular = specularStrength * spec * vec3(1.0);

  vec3 finalColor = ambient + diffuse + specular;

  gl_FragColor = vec4(finalColor, 1.0);
}