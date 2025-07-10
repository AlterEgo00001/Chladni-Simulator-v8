import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GPUComputationRenderer } from 'three/addons/misc/GPUComputationRenderer.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

const FDM_FRAGMENT_SHADER = `
  #version 300 es
  precision highp float;
  precision highp sampler2D;

  #define PI 3.141592653589793

  out vec4 out_FragColor;

  uniform sampler2D u_fdmTexture_read;
  uniform sampler2D u_modalPatternTexture;
  uniform float u_time;
  uniform float u_freq;
  uniform float u_excAmp;
  uniform float u_damp;
  uniform float u_dt;
  uniform float u_dx;
  uniform float u_D_flexural;
  uniform float u_rho_h;
  uniform float u_plateRadius;
  uniform int u_mParam;
  uniform int u_excMode;

  float sample_boundary(vec2 uv, vec2 offset) {
    vec2 sample_uv = uv + offset;
    vec2 sample_phys = (sample_uv - 0.5) * u_plateRadius * 2.0;
    if (length(sample_phys) > u_plateRadius) {
      return texture(u_fdmTexture_read, uv - offset).r;
    }
    return texture(u_fdmTexture_read, sample_uv).r;
  }

  void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(u_fdmTexture_read, 0));
    vec2 texelSize = 1.0 / vec2(textureSize(u_fdmTexture_read, 0));
    float physX = (uv.x - 0.5) * u_plateRadius * 2.0;
    float physY = (uv.y - 0.5) * u_plateRadius * 2.0;
    if (length(vec2(physX, physY)) > u_plateRadius) {
      out_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
      return;
    }
    vec4 data = texture(u_fdmTexture_read, uv);
    float u_curr = data.r;
    float u_prev = data.g;
    float inv_dx4 = 1.0 / (u_dx * u_dx * u_dx * u_dx);
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
    float excForce = 0.0;
    float timeSine = sin(2.0 * PI * u_freq * u_time);
    if (u_excMode == 0) {
      float theta = atan(physY, physX);
      float modalPattern = texture(u_modalPatternTexture, uv).r;
      excForce = u_excAmp * timeSine * modalPattern * cos(float(u_mParam) * theta);
    } else {
      vec2 centerUV = vec2(0.5, 0.5);
      float distSq = dot(uv - centerUV, uv - centerUV);
      if (distSq <= 0.0025) {
        excForce = u_excAmp * timeSine * exp(-distSq / 0.001);
      }
    }
    float K_coeff = (u_dt * u_dt * u_D_flexural) / u_rho_h;
    float F_coeff = (u_dt * u_dt) / u_rho_h;
    float u_next = (2.0 * u_curr - u_prev) - K_coeff * biharmonic + F_coeff * excForce;
    u_next *= (1.0 - u_damp);
    out_FragColor = vec4(u_next, u_curr, 0.0, 0.0);
  }
`;

const PARTICLE_PHYSICS_FRAGMENT_SHADER = `
  #version 300 es
  precision highp float;
  precision highp sampler2D;

  out vec4 out_FragColor;

  uniform sampler2D u_particleTexture_read;
  uniform sampler2D u_displacementTexture;
  uniform float u_plateRadius;
  uniform float u_plateWidth;
  uniform float u_dx;
  uniform float u_forceMult;
  uniform float u_damping;
  uniform float u_restitution;
  uniform float u_maxSpeed;
  uniform float u_deltaTime;
  uniform float u_repulsionRadius;
  uniform float u_repulsionStrength;
  uniform float u_stuckThreshold;
  void main() {
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(u_particleTexture_read, 0));
    vec4 data = texture(u_particleTexture_read, uv);
    vec2 pos = data.rg;
    vec2 vel = data.ba;
    if (pos.x > 900.0) {
      out_FragColor = vec4(pos, vel);
      return;
    }
    vec2 normPos = pos / u_plateWidth + 0.5;
    float disp = texture(u_displacementTexture, normPos).r;
    vec2 texelSizeDisp = 1.0 / vec2(textureSize(u_displacementTexture, 0));
    float gradX = (texture(u_displacementTexture, normPos + vec2(texelSizeDisp.x, 0.0)).r - texture(u_displacementTexture, normPos - vec2(texelSizeDisp.x, 0.0)).r) / (2.0 * u_dx);
    float gradY = (texture(u_displacementTexture, normPos + vec2(0.0, texelSizeDisp.y)).r - texture(u_displacementTexture, normPos - vec2(0.0, texelSizeDisp.y)).r) / (2.0 * u_dx);
    vec2 force = -2.0 * disp * vec2(gradX, gradY) * u_forceMult;
    if (u_repulsionStrength > 0.0 && u_repulsionRadius > 0.0) {
      vec2 texelSizeParticle = 1.0 / vec2(textureSize(u_particleTexture_read, 0));
      vec2 repulsionForce = vec2(0.0);
      const int checks = 4;
      for (int i = 1; i <= checks; i++) {
        float angle = float(i) / float(checks) * 6.283185;
        vec2 neighborUV = uv + vec2(cos(angle), sin(angle)) * texelSizeParticle;
        vec2 toNeighbor = texture(u_particleTexture_read, neighborUV).rg - pos;
        float distSq = dot(toNeighbor, toNeighbor);
        if (distSq < u_repulsionRadius * u_repulsionRadius && distSq > 0.00001) {
          float dist = sqrt(distSq);
          repulsionForce -= ((u_repulsionRadius - dist) / u_repulsionRadius) * normalize(toNeighbor) * u_repulsionStrength;
        }
      }
      force += repulsionForce;
    }
    vel = (vel + force * u_deltaTime) * u_damping;
    if (length(vel) > u_maxSpeed) vel = normalize(vel) * u_maxSpeed;
    pos += vel * u_deltaTime;
    if (length(pos) > u_plateRadius) {
      pos = normalize(pos) * u_plateRadius;
      vec2 normB = pos / u_plateRadius;
      vel -= (1.0 + u_restitution) * dot(vel, normB) * normB;
    }
    if (u_stuckThreshold > 0.0 && length(vel) < u_stuckThreshold && abs(disp) > 0.05) {
      pos = vec2(1001.0, 1001.0);
      vel = vec2(0.0, 0.0);
    }
    out_FragColor = vec4(pos, vel);
  }
`;

const PARTICLE_VERTEX_SHADER = `
  #version 300 es
  precision highp float;

  in float instanceId;
  in vec3 position;

  uniform mat4 modelViewMatrix;
  uniform mat4 projectionMatrix;
  uniform sampler2D u_particleTexture;
  uniform sampler2D u_displacementTexture;
  uniform vec2 u_particleTexResolution;
  uniform float u_plateWidth;
  uniform float u_visScale;
  uniform float u_maxVisAmp;
  uniform float u_rotationAngle;
  uniform float u_activeParticleCount;
  uniform float u_particleSize;

  out vec3 v_worldPosition;
  out vec3 v_normal;
  out vec2 v_particleUV;

  void main() {
    if (instanceId >= u_activeParticleCount) {
      gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
      return;
    }
    v_particleUV = vec2(
      mod(instanceId, u_particleTexResolution.x) + 0.5,
      floor(instanceId / u_particleTexResolution.x) + 0.5
    ) / u_particleTexResolution;
    vec4 data = texture(u_particleTexture, v_particleUV);
    vec2 pos2D = data.rg;
    if (pos2D.x > 900.0) {
      gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
      return;
    }
    vec2 normPos = pos2D / u_plateWidth + 0.5;
    float disp = texture(u_displacementTexture, normPos).r;
    float visHeight = clamp(disp * u_visScale, -u_maxVisAmp, u_maxVisAmp);
    float cosA = cos(u_rotationAngle);
    float sinA = sin(u_rotationAngle);
    float rotX = pos2D.x * cosA - pos2D.y * sinA;
    float rotZ = pos2D.x * sinA + pos2D.y * cosA;
    vec3 finalOffset = vec3(rotX, visHeight, rotZ);
    vec3 scaledPosition = position * u_particleSize;
    v_worldPosition = finalOffset + scaledPosition;
    v_normal = normalize(position);
    vec4 mvPosition = modelViewMatrix * vec4(v_worldPosition, 1.0);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const PARTICLE_FRAGMENT_SHADER = `
  #version 300 es
  precision highp float;

  in vec3 v_worldPosition;
  in vec3 v_normal;
  in vec2 v_particleUV;
  
  out vec4 out_FragColor;

  uniform sampler2D u_particleTexture;
  uniform vec3 u_lightPos;
  uniform vec3 u_cameraPos;
  uniform float u_colorMode;
  uniform vec3 u_globalColor;
  uniform float u_maxSpeedForColor;
  uniform vec3 u_coldColor;
  uniform vec3 u_hotColor;
  void main() {
    vec3 baseColor;
    if (u_colorMode > 0.5) {
      vec2 vel = texture(u_particleTexture, v_particleUV).ba;
      float speed = length(vel);
      float speedFactor = clamp(speed / u_maxSpeedForColor, 0.0, 1.0);
      baseColor = mix(u_coldColor, u_hotColor, speedFactor);
    } else {
      baseColor = u_globalColor;
    }
    vec3 normal = normalize(v_normal);
    vec3 lightDir = normalize(u_lightPos - v_worldPosition);
    vec3 viewDir = normalize(u_cameraPos - v_worldPosition);
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
    out_FragColor = vec4(finalColor, 1.0);
  }
`;

const PLATE_RADIUS_DEFAULT = 7.5;
const PLATE_THICKNESS_DEFAULT = 0.002;
const PLATE_DENSITY_DEFAULT = 7850;
const E_MODULUS_DEFAULT = 200e9;
const POISSON_RATIO_DEFAULT = 0.3;

const GPU_GRID_SIZE_DEFAULT = 255;
const GPU_FDM_STEPS_PER_FRAME_DEFAULT = 15;
const GPU_PARTICLE_COUNT_DEFAULT = 16384;

const PARTICLE_FORCE_BASE_DEFAULT = 2.5e6;
const PARTICLE_DAMPING_BASE_DEFAULT = 0.96;
const MAX_PARTICLE_SPEED_DEFAULT = 25.0;
const ENABLE_REPULSION_DEFAULT = true;
const REPULSION_RADIUS_DEFAULT = 0.18;
const REPULSION_STRENGTH_DEFAULT = 0.008;
const PARTICLE_RESTITUTION_DEFAULT = 0.4;
const STUCK_PARTICLE_THRESHOLD_DEFAULT = 0.02;

const MAX_VISUAL_AMPLITUDE_DEFAULT = 0.3;
const VISUAL_DEFORMATION_SCALE_DEFAULT = 50.0;
const VISUAL_PARTICLE_SIZE_DEFAULT = 0.045;

const EXC_BASE_AMP_DEFAULT = 2.0e4;
const EXC_LOW_CUTOFF_DEFAULT = 100;
const EXC_HIGH_CUTOFF_DEFAULT = 3000;
const EXC_MAX_FACTOR_DEFAULT = 3.0;
const EXC_MIN_FACTOR_DEFAULT = 0.5;

const ENABLE_SHADOWS_DEFAULT = false;
const ENABLE_STUCK_PARTICLE_CULLING_DEFAULT = true;
const ENABLE_COLOR_BY_SPEED_DEFAULT = true;
const ENABLE_DYNAMIC_PARTICLE_DENSITY_DEFAULT = false;
const MIN_DYNAMIC_PARTICLE_COUNT = 4096;
const PARTICLE_COUNT_UPDATE_THROTTLE_MS = 500;

const PITCH_MIN_FREQUENCY_HZ = 30;
const PITCH_MAX_FREQUENCY_HZ = 12000;
const PITCH_UPDATE_INTERVAL_SECONDS = 0.03;
const PITCH_CHANGE_THRESHOLD_HZ = 8.9;
const PITCH_SMOOTHING_FACTOR = 0.15;
const BPM_UPDATE_INTERVAL_SECONDS = 0.5;
const BPM_ANALYSIS_SAMPLE_RATE = 1.0 / PITCH_UPDATE_INTERVAL_SECONDS;

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const NOTE_TO_MIDI_NUMBER_OFFSET = { 'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11 };

class ChladniSimulator {
  constructor(besselRootsTable) {
    this.besselRootsTable = besselRootsTable;
    this.isReady = false;

    this.PLATE_RADIUS = PLATE_RADIUS_DEFAULT;
    this.PLATE_THICKNESS = PLATE_THICKNESS_DEFAULT;
    this.PLATE_DENSITY = PLATE_DENSITY_DEFAULT;
    this.E_MODULUS = E_MODULUS_DEFAULT;
    this.POISSON_RATIO = POISSON_RATIO_DEFAULT;
    this.D_FLEXURAL_RIGIDITY = 0;
    this.RHO_H_PLATE_SPECIFIC_DENSITY = 0;

    this.MAX_PARTICLE_COUNT = GPU_PARTICLE_COUNT_DEFAULT;
    this.ACTIVE_PARTICLE_COUNT = GPU_PARTICLE_COUNT_DEFAULT;
    this.PARTICLE_FORCE_BASE = PARTICLE_FORCE_BASE_DEFAULT;
    this.PARTICLE_DAMPING_BASE = PARTICLE_DAMPING_BASE_DEFAULT;
    this.MAX_PARTICLE_SPEED = MAX_PARTICLE_SPEED_DEFAULT;
    this.ENABLE_PARTICLE_REPULSION = ENABLE_REPULSION_DEFAULT;
    this.PARTICLE_REPULSION_RADIUS = REPULSION_RADIUS_DEFAULT;
    this.PARTICLE_REPULSION_STRENGTH = REPULSION_STRENGTH_DEFAULT;
    this.PARTICLE_BOUNDARY_RESTITUTION = PARTICLE_RESTITUTION_DEFAULT;
    this.STUCK_PARTICLE_THRESHOLD = STUCK_PARTICLE_THRESHOLD_DEFAULT;

    this.GRID_SIZE = GPU_GRID_SIZE_DEFAULT;
    this.FDM_STEPS_PER_FRAME = GPU_FDM_STEPS_PER_FRAME_DEFAULT;
    this.FDM_DAMPING_FACTOR = 0.000050;
    
    this.MAX_VISUAL_AMPLITUDE = MAX_VISUAL_AMPLITUDE_DEFAULT;
    this.VISUAL_DEFORMATION_SCALE = VISUAL_DEFORMATION_SCALE_DEFAULT;
    this.VISUAL_PARTICLE_SIZE = VISUAL_PARTICLE_SIZE_DEFAULT;
    this.EXCITATION_BASE_AMP = EXC_BASE_AMP_DEFAULT;
    this.EXC_LOW_CUTOFF = EXC_LOW_CUTOFF_DEFAULT;
    this.EXC_HIGH_CUTOFF = EXC_HIGH_CUTOFF_DEFAULT;
    this.EXC_MAX_FACTOR = EXC_MAX_FACTOR_DEFAULT;
    this.EXC_MIN_FACTOR = EXC_MIN_FACTOR_DEFAULT;

    this.enableShadows = ENABLE_SHADOWS_DEFAULT;
    this.enableStuckParticleCulling = ENABLE_STUCK_PARTICLE_CULLING_DEFAULT;
    this.enableColorBySpeed = ENABLE_COLOR_BY_SPEED_DEFAULT;
    this.enableDynamicParticleDensity = ENABLE_DYNAMIC_PARTICLE_DENSITY_DEFAULT;
    this.isSubtitlesEnabled = true;
    this.lastParticleCountUpdateTime = 0;

    this.plateRotationAngle = 0;
    this.plateRotationSpeed = 0;

    this.currentFrequency = 273;
    this.actualAppliedFrequency = 273;
    this.mParameter = 0;
    this.nParameter = 1;
    this.particleSimulationSpeedScale = 1.0;
    this.simulationTime = 0;
    this.areParticlesFrozen = false;
    this.drivingMechanism = 'modal';

    this.scene = null; this.camera = null; this.renderer = null; this.composer = null;
    this.orbitControls = null; this.particlesMesh = null;
    this.animationClock = new THREE.Clock();
    this.groundPlane = null; this.dirLight1 = null; this.dirLight2 = null;

    this.gpuCompute = null;
    this.fdmVariable = null;
    this.particleVariable = null;
    this.particleMeshMaterial = null;
    this.modalPatternTexture = null;

    this.mainAudioContext = null; this.generatedSoundOscillator = null; this.generatedSoundGainNode = null;
    this.isGeneratedSoundEnabled = false;
    
    this.playlistFiles = [];
    this.currentPlaylistIndex = -1;
    this.audioElement = this._createAudioElement();
    this.audioFileSourceNode = null;
    this.isAudioFilePlaying = false;
    this.isAudioFilePaused = false;
    this.audioFileDuration = 0;

    this.subtitleContainer = null;
    this.currentSubtitles = [];
    this.activeFetchID = 0;
    
    this.pitchDetectorAnalyserNode = null;
    this.pitchDetectorSignalBuffer = null;
    this.fftAnalyserNode = null;
    this.frequencyData = null;
    this.lastPitchUpdateTime = 0;
    this.smoothedPitchFrequency = 273;
    
    this.onsetHistory = [];
    this.onsetHistoryMaxLength = 300;
    this.lastBPM = 0;
    this.lastBPMUpdateTime = 0;
    
    this.isMicrophoneEnabled = false;
    this.microphoneStream = null;
    this.microphoneSourceNode = null;
    this.isDesktopAudioEnabled = false;
    this.desktopStream = null;
    this.desktopAudioSourceNode = null;

    this.currentPianoOctave = 4;
    this.activePianoKeys = new Set();
    this.keyboardPressedKeys = new Set();
    this.keyToNoteMapping = {
      'KeyA': 'C', 'KeyS': 'D', 'KeyD': 'E', 'KeyF': 'F',
      'KeyG': 'G', 'KeyH': 'A', 'KeyJ': 'B', 'KeyK': 'C5',
      'KeyW': 'C#', 'KeyE': 'D#', 'KeyT': 'F#', 'KeyY': 'G#', 'KeyU': 'A#'
    };

    this.uiElements = {};
    this.defaultAdvancedSettings = {};
    this.besselZerosCache = {};
    
    try {
        this._mainInitialization();
    } catch (error) {
        console.error("Critical error during main initialization:", error);
        document.body.innerHTML = `<div style="color: #e06c75; background-color:#282c34; padding: 20px; text-align: center; font-family: sans-serif;"><h1>Критическая ошибка</h1><p>Не удалось инициализировать симулятор. Проверьте консоль разработчика (F12) для получения подробной информации.</p><p style="margin-top: 10px; color: #abb2bf;">${error.message}</p></div>`;
    }
  }

  _roundToOddInteger(number) {
    number = Math.max(1, Math.round(number));
    return (number % 2 === 0) ? number + 1 : number;
  }

  _parseInputNumber(value, defaultValue = 0, isInt = false, minVal = -Infinity, maxVal = Infinity) {
    let num = isInt ? parseInt(value) : parseFloat(value);
    if (isNaN(num) || !isFinite(num)) return defaultValue;
    return Math.max(minVal, Math.min(maxVal, num));
  }

  _formatTimeMMSS(totalSeconds) {
    if (isNaN(totalSeconds) || totalSeconds < 0) return "0:00";
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = Math.floor(totalSeconds % 60).toString().padStart(2, '0');
    return `${minutes}:${seconds}`;
  }
  
  _besselJ_lib(order, xValue) {
    if (typeof BESSEL === 'undefined' || typeof BESSEL.besselj !== 'function') {
      console.warn("Bessel library not found.");
      return 0;
    }
    return BESSEL.besselj(parseFloat(xValue), parseFloat(order));
  }

  _getBesselZero(mOrder, nthRootOneIndexed) {
    const m = Math.round(mOrder);
    const n = Math.round(nthRootOneIndexed);
    if (m < 0 || n <= 0) return null;
    const cacheKey = `${m},${n}`;
    if (this.besselZerosCache[cacheKey]) return this.besselZerosCache[cacheKey];
    if (!this.besselRootsTable || !this.besselRootsTable.hasOwnProperty(m) || (n - 1) >= this.besselRootsTable[m].length) return null;
    const root = this.besselRootsTable[m][n - 1];
    this.besselZerosCache[cacheKey] = root;
    return root;
  }

  _updatePhysicalConstants() {
    this.D_FLEXURAL_RIGIDITY = (this.E_MODULUS * Math.pow(this.PLATE_THICKNESS, 3)) / (12 * (1 - Math.pow(this.POISSON_RATIO, 2)));
    this.RHO_H_PLATE_SPECIFIC_DENSITY = this.PLATE_DENSITY * this.PLATE_THICKNESS;
  }
  
  _getResonantFrequencyKirchhoff(mVal, nVal) {
    const lambda_mn = this._getBesselZero(mVal, nVal);
    if (lambda_mn === null || lambda_mn <= 0) return 20;
    this._updatePhysicalConstants();
    const freq = (Math.pow(lambda_mn / this.PLATE_RADIUS, 2) / (2 * Math.PI)) * Math.sqrt(this.D_FLEXURAL_RIGIDITY / this.RHO_H_PLATE_SPECIFIC_DENSITY);
    return Math.max(1, (isFinite(freq) ? freq : 20));
  }
  
  _mainInitialization() {
    this._mapUIElements();
    this._storeDefaultSimulationSettings();
    this._setupThreeJSScene();
    this._setupWebAudioSystem();
    this._setupGPUSimulation();
    this._createParticleSystem();
    this._setupEventListeners();
    this._createPianoKeys();
    this._resetAllSettingsToDefaults(false);

    const gpuWarning = document.createElement('p'); 
    gpuWarning.textContent = 'Симуляция на GPU (WebGL 2.0)'; 
    gpuWarning.style.cssText = 'color: #98c379; font-size: 12px; text-align: center; margin: 5px 0; padding: 3px; border: 1px solid #98c379; border-radius: 4px;';
    if (this.uiElements.controls) {
      const fs = this.uiElements.controls.querySelector('fieldset');
      if (fs) this.uiElements.controls.insertBefore(gpuWarning, fs);
    }

    this.isReady = true;
    this.animationClock.start();
    this._animateScene();
  }

  _setupGPUSimulation() {
    if (!this.renderer.capabilities.isWebGL2) {
      throw new Error("Fatal Error: This application requires WebGL 2.0. Your browser does not support it.");
    }
    
    const particleTexSize = Math.ceil(Math.sqrt(this.MAX_PARTICLE_COUNT));
    const fdmTexSize = this.GRID_SIZE;
    const computeTexSize = Math.max(particleTexSize, fdmTexSize);
    
    this.gpuCompute = new GPUComputationRenderer(computeTexSize, computeTexSize, this.renderer);
    
    const fdmTexture = this.gpuCompute.createTexture();
    this._fillFDMTexture(fdmTexture, fdmTexSize);
    this.fdmVariable = this.gpuCompute.addVariable("u_fdmTexture_read", FDM_FRAGMENT_SHADER, fdmTexture);
    if (!this.fdmVariable) throw new Error("Failed to create FDM GPGPU variable. Check FDM shader for errors.");
    
    const particleTexture = this.gpuCompute.createTexture();
    this._fillParticleTexture(particleTexture);
    this.particleVariable = this.gpuCompute.addVariable("u_particleTexture_read", PARTICLE_PHYSICS_FRAGMENT_SHADER, particleTexture);
    if (!this.particleVariable) throw new Error("Failed to create Particle GPGPU variable. Check Particle Physics shader for errors.");

    this.gpuCompute.setVariableDependencies(this.fdmVariable, [this.fdmVariable]);
    this.gpuCompute.setVariableDependencies(this.particleVariable, [this.particleVariable, this.fdmVariable]);
    
    this.fdmVariable.material.uniforms['u_modalPatternTexture'] = { value: null };
    this.fdmVariable.material.uniforms['u_time'] = { value: 0.0 };
    this.fdmVariable.material.uniforms['u_freq'] = { value: 0.0 };
    this.fdmVariable.material.uniforms['u_excAmp'] = { value: 0.0 };
    this.fdmVariable.material.uniforms['u_damp'] = { value: 0.0 };
    this.fdmVariable.material.uniforms['u_dt'] = { value: 0.0 };
    this.fdmVariable.material.uniforms['u_dx'] = { value: 0.0 };
    this.fdmVariable.material.uniforms['u_D_flexural'] = { value: 0.0 };
    this.fdmVariable.material.uniforms['u_rho_h'] = { value: 0.0 };
    this.fdmVariable.material.uniforms['u_plateRadius'] = { value: this.PLATE_RADIUS };
    this.fdmVariable.material.uniforms['u_mParam'] = { value: 0 };
    this.fdmVariable.material.uniforms['u_excMode'] = { value: 0 };

    this.particleVariable.material.uniforms['u_displacementTexture'] = { value: null };
    this.particleVariable.material.uniforms['u_plateRadius'] = { value: this.PLATE_RADIUS };
    this.particleVariable.material.uniforms['u_plateWidth'] = { value: this.PLATE_RADIUS * 2.0 };
    this.particleVariable.material.uniforms['u_dx'] = { value: 0.0 };
    this.particleVariable.material.uniforms['u_forceMult'] = { value: 0.0 };
    this.particleVariable.material.uniforms['u_damping'] = { value: 0.0 };
    this.particleVariable.material.uniforms['u_restitution'] = { value: 0.0 };
    this.particleVariable.material.uniforms['u_maxSpeed'] = { value: 0.0 };
    this.particleVariable.material.uniforms['u_deltaTime'] = { value: 0.0 };
    this.particleVariable.material.uniforms['u_repulsionRadius'] = { value: 0.0 };
    this.particleVariable.material.uniforms['u_repulsionStrength'] = { value: 0.0 };
    this.particleVariable.material.uniforms['u_stuckThreshold'] = { value: 0.0 };

    const error = this.gpuCompute.init();
    if (error !== null) {
      throw new Error(`GPUComputationRenderer Error: ${error}`);
    }
  }

  _reinitGPUSimulation() {
    if (this.gpuCompute) {
        Object.values(this.gpuCompute.renderTargets).forEach(rt => rt.dispose());
        this.gpuCompute.dispose();
    }
    this._setupGPUSimulation();
    this._createParticleSystem();
    this._resetFullSimulationState();
  }

  _fillFDMTexture(texture, size) {
    const arr = texture.image.data;
    texture.image.width = size;
    texture.image.height = size;
    for (let i = 0; i < arr.length; i += 4) {
      arr[i] = 0.0; arr[i + 1] = 0.0; arr[i + 2] = 0.0; arr[i + 3] = 0.0;
    }
  }

  _fillParticleTexture(texture) {
    const arr = texture.image.data;
    for (let i = 0; i < arr.length; i += 4) {
      const r = Math.sqrt(Math.random()) * this.PLATE_RADIUS;
      const angle = Math.random() * 2 * Math.PI;
      arr[i] = r * Math.cos(angle); 
      arr[i + 1] = r * Math.sin(angle); 
      arr[i + 2] = 0.0; 
      arr[i + 3] = 0.0; 
    }
  }

  _updateModalPatternTexture() {
    const size = this.GRID_SIZE;
    const data = new Float32Array(size * size);
    const b_zero = this._getBesselZero(this.mParameter, this.nParameter);
    if (b_zero !== null && b_zero > 0) {
      const k_mn = b_zero / this.PLATE_RADIUS;
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          const x = (j / (size - 1.0) - 0.5) * this.PLATE_RADIUS * 2.0;
          const y = (i / (size - 1.0) - 0.5) * this.PLATE_RADIUS * 2.0;
          const r_phys = Math.hypot(x, y);
          const index = i * size + j;
          data[index] = (r_phys <= this.PLATE_RADIUS) ? this._besselJ_lib(this.mParameter, k_mn * r_phys) : 0.0;
        }
      }
    }
    if (this.modalPatternTexture) this.modalPatternTexture.dispose();
    this.modalPatternTexture = new THREE.DataTexture(data, size, size, THREE.RedFormat, THREE.FloatType);
    this.modalPatternTexture.needsUpdate = true;
  }
  
  _setupThreeJSScene() {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x000000);
    this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    this.camera.position.set(0, this.PLATE_RADIUS * 1.6, this.PLATE_RADIUS * 2.0);
    this.camera.lookAt(0, 0, 0);
    
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('webgl2', { antialias: true, powerPreference: "high-performance" });
    if (!context) {
        throw new Error('WebGL 2.0 is not available on this device.');
    }
    document.getElementById('scene-container').appendChild(canvas);

    this.renderer = new THREE.WebGLRenderer({ canvas: canvas, context: context });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = this.enableShadows;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    this.orbitControls = new OrbitControls(this.camera, this.renderer.domElement);
    this.orbitControls.enableDamping = true;
    this.orbitControls.dampingFactor = 0.04;
    this.orbitControls.minDistance = this.PLATE_RADIUS * 0.3;
    this.orbitControls.maxDistance = this.PLATE_RADIUS * 12;
    this.orbitControls.target.set(0, 0, 0);
    this.orbitControls.autoRotate = false;
    this.orbitControls.enablePan = false;
    
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    
    this.dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    this.dirLight1.position.set(this.PLATE_RADIUS * 2, this.PLATE_RADIUS * 3, this.PLATE_RADIUS * 1.5);
    this.dirLight1.castShadow = this.enableShadows;
    this.dirLight1.shadow.mapSize.set(1024, 1024);
    this.dirLight1.shadow.camera.near = 0.5;
    this.dirLight1.shadow.camera.far = this.PLATE_RADIUS * 10;
    this.scene.add(this.dirLight1);
    
    this.dirLight2 = new THREE.DirectionalLight(0xffddaa, 0.5);
    this.dirLight2.position.set(-this.PLATE_RADIUS * 2, this.PLATE_RADIUS * 1, -this.PLATE_RADIUS * 1.5);
    this.scene.add(this.dirLight2);

    const groundGeometry = new THREE.PlaneGeometry(this.PLATE_RADIUS * 4, this.PLATE_RADIUS * 4);
    const groundMaterial = new THREE.ShadowMaterial({ opacity: 0.3 });
    this.groundPlane = new THREE.Mesh(groundGeometry, groundMaterial);
    this.groundPlane.rotation.x = -Math.PI / 2;
    this.groundPlane.position.y = -0.05;
    this.groundPlane.receiveShadow = this.enableShadows;
    this.groundPlane.visible = this.enableShadows;
    this.scene.add(this.groundPlane);

    const renderScene = new RenderPass(this.scene, this.camera);
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.0, 0.4, 0.85);
    bloomPass.threshold = 0.1;
    bloomPass.strength = 0.3;
    bloomPass.radius = 0.3;
    
    const renderTarget = new THREE.WebGLRenderTarget(window.innerWidth, window.innerHeight, {
        type: THREE.HalfFloatType,
        format: THREE.RGBAFormat,
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
    });
    
    this.composer = new EffectComposer(this.renderer, renderTarget);
    this.composer.addPass(renderScene);
    this.composer.addPass(bloomPass);
  }

  _createParticleSystem() {
    if (this.particlesMesh) {
      this.scene.remove(this.particlesMesh);
      if(this.particlesMesh.geometry) this.particlesMesh.geometry.dispose();
      if(this.particlesMesh.material) this.particlesMesh.material.dispose();
    }

    const particleCount = this.MAX_PARTICLE_COUNT;
    const geometry = new THREE.InstancedBufferGeometry();
    const baseGeom = new THREE.SphereGeometry(1, 6, 4);
    geometry.index = baseGeom.index;
    geometry.attributes.position = baseGeom.attributes.position;
    baseGeom.dispose();

    const instanceIds = new Float32Array(particleCount);
    for (let i = 0; i < particleCount; i++) instanceIds[i] = i;
    geometry.setAttribute('instanceId', new THREE.InstancedBufferAttribute(instanceIds, 1));

    const particleTexRes = Math.ceil(Math.sqrt(particleCount));
    this.particleMeshMaterial = new THREE.ShaderMaterial({
      vertexShader: PARTICLE_VERTEX_SHADER,
      fragmentShader: PARTICLE_FRAGMENT_SHADER,
      uniforms: {
        u_particleTexture: { value: null },
        u_displacementTexture: { value: null },
        u_particleTexResolution: { value: new THREE.Vector2(particleTexRes, particleTexRes) },
        u_plateWidth: { value: this.PLATE_RADIUS * 2.0 },
        u_visScale: { value: this.VISUAL_DEFORMATION_SCALE },
        u_maxVisAmp: { value: this.MAX_VISUAL_AMPLITUDE },
        u_rotationAngle: { value: 0.0 },
        u_activeParticleCount: { value: this.ACTIVE_PARTICLE_COUNT },
        u_particleSize: { value: this.VISUAL_PARTICLE_SIZE },
        u_lightPos: { value: this.dirLight1.position },
        u_cameraPos: { value: this.camera.position },
        u_colorMode: { value: this.enableColorBySpeed ? 1.0 : 0.0 },
        u_globalColor: { value: new THREE.Color(0x00ddff) },
        u_maxSpeedForColor: { value: this.MAX_PARTICLE_SPEED },
        u_coldColor: { value: new THREE.Color(0x0055ff) },
        u_hotColor: { value: new THREE.Color(0xffff88) }
      }
    });

    this.particlesMesh = new THREE.Mesh(geometry, this.particleMeshMaterial);
    this.particlesMesh.castShadow = this.enableShadows;
    this.scene.add(this.particlesMesh);
  }

  _resetFullSimulationState() {
    this._updatePhysicalConstants();
    this._setupPlateParametersForCurrentMode();
    this._updateModalPatternTexture();

    const fdmTexture = this.gpuCompute.createTexture();
    this._fillFDMTexture(fdmTexture, this.GRID_SIZE);
    this.gpuCompute.renderTexture(fdmTexture, this.fdmVariable.renderTargets[0]);
    this.gpuCompute.renderTexture(fdmTexture, this.fdmVariable.renderTargets[1]);
    fdmTexture.dispose();
    
    const particleTexture = this.gpuCompute.createTexture();
    this._fillParticleTexture(particleTexture);
    this.gpuCompute.renderTexture(particleTexture, this.particleVariable.renderTargets[0]);
    this.gpuCompute.renderTexture(particleTexture, this.particleVariable.renderTargets[1]);
    particleTexture.dispose();
  }

  _getFrequencyDependentExcitationAmplitude(freq) { 
    const base = this.EXCITATION_BASE_AMP; 
    if (freq <= 0 || !isFinite(freq)) return base * this.EXC_MAX_FACTOR; 
    if (freq < this.EXC_LOW_CUTOFF) return base * this.EXC_MAX_FACTOR; 
    if (freq < this.EXC_HIGH_CUTOFF) { 
      const factorRange = this.EXC_MAX_FACTOR - this.EXC_MIN_FACTOR; 
      const freqRange = this.EXC_HIGH_CUTOFF - this.EXC_LOW_CUTOFF; 
      if (freqRange <= 0) return base * this.EXC_MAX_FACTOR; 
      return base * (this.EXC_MAX_FACTOR - factorRange * ((freq - this.EXC_LOW_CUTOFF) / freqRange)); 
    }
    return base * this.EXC_MIN_FACTOR; 
  }

  _animateScene() {
    requestAnimationFrame(this._animateScene.bind(this));
    if (!this.isReady) return;

    const deltaTime = Math.min(this.animationClock.getDelta(), 0.05);
    this.simulationTime += deltaTime;
    
    this.orbitControls.update();
    
    if (this.plateRotationSpeed !== 0) {
      this.plateRotationAngle = (this.plateRotationAngle + this.plateRotationSpeed * 2 * Math.PI * deltaTime) % (2 * Math.PI);
    }

    if (!this.areParticlesFrozen && this.gpuCompute) {
      this._updatePhysicalConstants();
      const dx = (this.PLATE_RADIUS * 2.0) / (this.GRID_SIZE - 1);
      const stabilityFactor = 0.08;
      const dt_simulation_step = (stabilityFactor * dx * dx * Math.sqrt(this.RHO_H_PLATE_SPECIFIC_DENSITY / this.D_FLEXURAL_RIGIDITY));
      
      this.fdmVariable.material.uniforms.u_dt.value = dt_simulation_step * this.particleSimulationSpeedScale * this.FDM_STEPS_PER_FRAME;
      this.fdmVariable.material.uniforms.u_dx.value = dx;
      this.fdmVariable.material.uniforms.u_D_flexural.value = this.D_FLEXURAL_RIGIDITY;
      this.fdmVariable.material.uniforms.u_rho_h.value = this.RHO_H_PLATE_SPECIFIC_DENSITY;
      this.fdmVariable.material.uniforms.u_freq.value = this.actualAppliedFrequency;
      this.fdmVariable.material.uniforms.u_damp.value = this.FDM_DAMPING_FACTOR;
      this.fdmVariable.material.uniforms.u_excAmp.value = this._getFrequencyDependentExcitationAmplitude(this.actualAppliedFrequency);
      this.fdmVariable.material.uniforms.u_time.value = this.simulationTime;
      this.fdmVariable.material.uniforms.u_mParam.value = this.mParameter;
      this.fdmVariable.material.uniforms.u_modalPatternTexture.value = this.modalPatternTexture;
      
      this.particleVariable.material.uniforms.u_deltaTime.value = deltaTime * this.particleSimulationSpeedScale;
      this.particleVariable.material.uniforms.u_dx.value = dx;
      this.particleVariable.material.uniforms.u_forceMult.value = this.PARTICLE_FORCE_BASE;
      this.particleVariable.material.uniforms.u_damping.value = this.PARTICLE_DAMPING_BASE;
      this.particleVariable.material.uniforms.u_restitution.value = this.PARTICLE_BOUNDARY_RESTITUTION;
      this.particleVariable.material.uniforms.u_maxSpeed.value = this.MAX_PARTICLE_SPEED;
      this.particleVariable.material.uniforms.u_repulsionRadius.value = this.ENABLE_PARTICLE_REPULSION ? this.PARTICLE_REPULSION_RADIUS : 0.0;
      this.particleVariable.material.uniforms.u_repulsionStrength.value = this.PARTICLE_REPULSION_STRENGTH;
      this.particleVariable.material.uniforms.u_stuckThreshold.value = this.enableStuckParticleCulling ? this.STUCK_PARTICLE_THRESHOLD : -1.0;
      
      this.gpuCompute.compute();
      
      this.particleVariable.material.uniforms.u_displacementTexture.value = this.gpuCompute.getCurrentRenderTarget(this.fdmVariable).texture;
    }
    
    if (this.particleMeshMaterial) {
      const uniforms = this.particleMeshMaterial.uniforms;
      uniforms.u_particleTexture.value = this.gpuCompute.getCurrentRenderTarget(this.particleVariable).texture;
      uniforms.u_displacementTexture.value = this.gpuCompute.getCurrentRenderTarget(this.fdmVariable).texture;
      uniforms.u_rotationAngle.value = this.plateRotationAngle;
      
      const freq = this.actualAppliedFrequency || this.currentFrequency;
      const hue = THREE.MathUtils.clamp((Math.log10(Math.max(20, freq)) - Math.log10(20)) / (Math.log10(20000) - Math.log10(20) + 1e-9), 0, 1) * 0.7;
      uniforms.u_globalColor.value.setHSL(hue, 0.98, 0.58);
      uniforms.u_activeParticleCount.value = this.enableDynamicParticleDensity ? this.ACTIVE_PARTICLE_COUNT : this.MAX_PARTICLE_COUNT;
      uniforms.u_particleSize.value = this.VISUAL_PARTICLE_SIZE;
    }
    
    this._updateAudioProcessing();
    
    if (this.mainAudioContext?.state === 'running' && this.simulationTime - this.lastBPMUpdateTime > BPM_UPDATE_INTERVAL_SECONDS) {
      this._updateBPM();
      this.lastBPMUpdateTime = this.simulationTime;
    }
    
    this.composer.render(deltaTime);
  }

  _updateAudioProcessing() {
    const isAudioActive = (this.drivingMechanism === 'audio' && this.isAudioFilePlaying && !this.isAudioFilePaused) ||
                          (this.drivingMechanism === 'microphone' && this.isMicrophoneEnabled) ||
                          (this.drivingMechanism === 'desktop_audio' && this.isDesktopAudioEnabled);

    if (isAudioActive && this.pitchDetectorAnalyserNode && this.mainAudioContext?.state === 'running') {
      if (this.mainAudioContext.currentTime - this.lastPitchUpdateTime > PITCH_UPDATE_INTERVAL_SECONDS) {
        this.lastPitchUpdateTime = this.mainAudioContext.currentTime;
        if (this.pitchDetectorSignalBuffer) {
          try {
            this.pitchDetectorAnalyserNode.getFloatTimeDomainData(this.pitchDetectorSignalBuffer);
            const detectedFreq = this._autoCorrelatePitch(this.pitchDetectorSignalBuffer, this.mainAudioContext.sampleRate);
            if (detectedFreq !== -1) {
              this.smoothedPitchFrequency = THREE.MathUtils.lerp(this.smoothedPitchFrequency, detectedFreq, PITCH_SMOOTHING_FACTOR);
              if (Math.abs(this.actualAppliedFrequency - this.smoothedPitchFrequency) > PITCH_CHANGE_THRESHOLD_HZ) {
                this.actualAppliedFrequency = this.smoothedPitchFrequency;
                this._resetFullSimulationState();
              } else {
                this.actualAppliedFrequency = this.smoothedPitchFrequency;
              }
              this.currentFrequency = detectedFreq;
              this._updateFrequencyControlsUI();
              this._updatePitchDetectorUI(detectedFreq);
            }
          } catch (e) { console.warn("Error processing pitch data:", e); }
        }
      }

      if (this.fftAnalyserNode) {
        const timeDomainData = new Float32Array(this.fftAnalyserNode.fftSize);
        this.fftAnalyserNode.getFloatTimeDomainData(timeDomainData);
        let sumSq = 0;
        for (let i = 0; i < timeDomainData.length; i++) sumSq += timeDomainData[i] * timeDomainData[i];
        this.onsetHistory.push(Math.sqrt(sumSq / timeDomainData.length));
        if (this.onsetHistory.length > this.onsetHistoryMaxLength) this.onsetHistory.shift();
      }
    }

    if (this.drivingMechanism === 'audio') {
      this._updateAudioFileProgressControlsUI();
      if (this.isSubtitlesEnabled) this._updateSubtitles();
    }
    
    if (this.enableDynamicParticleDensity && this.mainAudioContext?.state === 'running') {
      this._updateDynamicParticleDensity();
    }
  }

  _updatePitchDetectorUI(freq) {
    if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'block';
    if (this.uiElements.pitch) this.uiElements.pitch.innerText = Math.round(freq);
    let midi = Math.round(12 * (Math.log(freq / 440) / Math.log(2))) + 69;
    if (this.uiElements.note) this.uiElements.note.innerText = NOTE_NAMES_SHARP[midi % 12];
    let cents = Math.floor(1200 * Math.log(freq / (440 * Math.pow(2, (midi - 69) / 12))) / Math.log(2));
    if (this.uiElements.detune_amt) this.uiElements.detune_amt.innerText = Math.abs(cents);
    if (this.uiElements.detune) this.uiElements.detune.className = cents === 0 ? "" : (cents < 0 ? "flat" : "sharp");
  }

  _autoCorrelatePitch(buffer, sampleRate) {
    let bufSize = buffer.length;
    let rms = 0;
    for (let i = 0; i < bufSize; i++) rms += buffer[i] * buffer[i];
    rms = Math.sqrt(rms / bufSize);
    if (rms < 0.01) return -1;

    let C = new Float32Array(bufSize).fill(0);
    for (let lag = 0; lag < bufSize; lag++) {
      let sum = 0;
      for (let i = 0; i < bufSize - lag; i++) sum += buffer[i] * buffer[i + lag];
      C[lag] = sum;
    }

    let d = 0;
    while (d < bufSize && C[d] > C[d + 1]) d++;
    
    let maxval = -1, maxpos = -1;
    for (let i = d; i < bufSize; i++) {
      if (C[i] > maxval) { maxval = C[i]; maxpos = i; }
    }
    
    let T0 = maxpos;
    if (T0 > 0 && T0 < bufSize - 1) {
        let y1 = C[T0 - 1], y2 = C[T0], y3 = C[T0 + 1];
        let p = (y3 - y1) / (2 * (2 * y2 - y3 - y1));
        if (isFinite(p)) T0 += p;
    }

    if (T0 === 0) return -1;
    const freq = sampleRate / T0;
    return (freq < PITCH_MIN_FREQUENCY_HZ || freq > PITCH_MAX_FREQUENCY_HZ) ? -1 : freq;
  }
  
  _setupWebAudioSystem() {
    try {
      this.mainAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      const resume = () => { if (this.mainAudioContext?.state === 'suspended') this.mainAudioContext.resume(); };
      document.addEventListener('click', resume, { once: true });
      this.pitchDetectorAnalyserNode = this.mainAudioContext.createAnalyser();
      this.pitchDetectorAnalyserNode.fftSize = 2048;
      this.pitchDetectorSignalBuffer = new Float32Array(this.pitchDetectorAnalyserNode.fftSize);
      
      this.fftAnalyserNode = this.mainAudioContext.createAnalyser();
      this.fftAnalyserNode.fftSize = 256;
      this.frequencyData = new Uint8Array(this.fftAnalyserNode.frequencyBinCount);

    } catch (e) {
      throw new Error("Web Audio API is not available in this browser.");
    }
  }
  
  _updateDynamicParticleDensity() {
    if (!this.fftAnalyserNode || this.mainAudioContext.state !== 'running') return;
    
    const currentTime = this.mainAudioContext.currentTime;
    if (currentTime - this.lastParticleCountUpdateTime < (PARTICLE_COUNT_UPDATE_THROTTLE_MS / 1000.0)) return;
    this.lastParticleCountUpdateTime = currentTime;

    this.fftAnalyserNode.getByteFrequencyData(this.frequencyData);
    let sumVol = 0;
    for (let i = 0; i < this.frequencyData.length; i++) sumVol += this.frequencyData[i];
    const avgVol = this.frequencyData.length > 0 ? sumVol / this.frequencyData.length : 0;
    const volumeFactor = THREE.MathUtils.clamp(avgVol / 128.0, 0.05, 1.5);
    
    const newParticleCount = MIN_DYNAMIC_PARTICLE_COUNT + (this.MAX_PARTICLE_COUNT - MIN_DYNAMIC_PARTICLE_COUNT) * volumeFactor;
    this.ACTIVE_PARTICLE_COUNT = Math.round(THREE.MathUtils.clamp(newParticleCount, MIN_DYNAMIC_PARTICLE_COUNT, this.MAX_PARTICLE_COUNT));
  }
  
  _setupPlateParametersForCurrentMode() {
    if (this.drivingMechanism === 'modal') {
      this.currentFrequency = this._getResonantFrequencyKirchhoff(this.mParameter, this.nParameter);
      this.actualAppliedFrequency = this.currentFrequency;
      if (this.fdmVariable) this.fdmVariable.material.uniforms.u_excMode.value = 0;
    } else {
      this.actualAppliedFrequency = this.currentFrequency;
      if (this.fdmVariable) this.fdmVariable.material.uniforms.u_excMode.value = 1;
    }
    
    if (this.uiElements.particleSpeedSlider) {
      const v = parseFloat(this.uiElements.particleSpeedSlider.value);
      if (v <= 50) this.particleSimulationSpeedScale = 0.1 + (1.0 - 0.1) * (v / 50);
      else this.particleSimulationSpeedScale = 1.0 + (10.0 - 1.0) * ((v - 50) / 50);
      if (this.uiElements.speedValueText) this.uiElements.speedValueText.textContent = `${this.particleSimulationSpeedScale.toFixed(2)}x`;
    }
    
    this._updateFrequencyControlsUI();
  }

  _setActiveDrivingMechanism(newMechanism) {
    this.drivingMechanism = newMechanism;
    if (this.uiElements.activeModeDisplay) {
        const modeMap = {'modal': 'Режим: Модальный', 'point': 'Режим: Точка', 'audio': 'Режим: Аудиофайл', 'microphone': 'Режим: Микрофон', 'desktop_audio': 'Режим: Захват звука'};
        this.uiElements.activeModeDisplay.innerHTML = modeMap[newMechanism] || 'Режим: Неизвестный';
    }
    this._resetFullSimulationState();
  }
  
  _resetAllSettingsToDefaults(isAdvancedOnly = false) {
    const prevPC = this.MAX_PARTICLE_COUNT;
    const prevGS = this.GRID_SIZE;
    
    Object.keys(this.defaultAdvancedSettings).forEach(k => this[k] = this.defaultAdvancedSettings[k]);
    this.GRID_SIZE = this._roundToOddInteger(this.defaultAdvancedSettings.GRID_SIZE);
    this.MAX_PARTICLE_COUNT = this.defaultAdvancedSettings.MAX_PARTICLE_COUNT;
    this.ACTIVE_PARTICLE_COUNT = this.enableDynamicParticleDensity ? MIN_DYNAMIC_PARTICLE_COUNT : this.MAX_PARTICLE_COUNT;

    if (!isAdvancedOnly) {
      this.enableShadows = ENABLE_SHADOWS_DEFAULT;
      this.enableStuckParticleCulling = ENABLE_STUCK_PARTICLE_CULLING_DEFAULT;
      this.enableColorBySpeed = ENABLE_COLOR_BY_SPEED_DEFAULT;
      this.enableDynamicParticleDensity = ENABLE_DYNAMIC_PARTICLE_DENSITY_DEFAULT;
      this.isSubtitlesEnabled = true;
      
      this.mParameter = 0; this.nParameter = 1; this.currentFrequency = 273;
      this.particleSimulationSpeedScale = 1.0;
      if (this.uiElements.particleSpeedSlider) this.uiElements.particleSpeedSlider.value = 50;
      this.plateRotationSpeed = 0;
      if (this.uiElements.plateRotationSpeedSlider) this.uiElements.plateRotationSpeedSlider.value = 0;
      
      Object.keys(this.besselZerosCache).forEach(key => delete this.besselZerosCache[key]);
      
      if (this.isMicrophoneEnabled) this._toggleMicrophoneInput();
      if (this.isDesktopAudioEnabled) this._toggleDesktopAudio();
      this._stopLoadedAudioFilePlayback(true);
      if (this.isGeneratedSoundEnabled) this._toggleGeneratedSoundPlayback(true);
      
      this._setActiveDrivingMechanism('modal');
    }
    
    if (prevPC !== this.MAX_PARTICLE_COUNT || prevGS !== this.GRID_SIZE) {
      this._reinitGPUSimulation();
    } else {
      this._createParticleSystem();
      this._resetFullSimulationState();
    }
    
    this._populateAdvancedControlsUI();
    this._updateUIToggleButtons();
    this._applyShadowSettings();
  }

  _applyAdvancedSettingChange(paramName, value) {
    const isCheckbox = typeof value === 'boolean';
    const isInt = paramName.includes("Count") || paramName.includes("Steps") || paramName.includes("Grid");
    const newVal = isCheckbox ? value : this._parseInputNumber(value, this[paramName], isInt);
    
    if (this[paramName] === newVal && !isCheckbox) return;

    let needsGpuReinit = false;
    
    if (paramName === 'GRID_SIZE') {
        const roundedVal = this._roundToOddInteger(newVal);
        if (this.GRID_SIZE !== roundedVal) { this.GRID_SIZE = roundedVal; needsGpuReinit = true; }
    } else if (paramName === 'MAX_PARTICLE_COUNT') {
        if (this.MAX_PARTICLE_COUNT !== newVal) { 
            this.MAX_PARTICLE_COUNT = newVal; 
            this.ACTIVE_PARTICLE_COUNT = this.enableDynamicParticleDensity ? MIN_DYNAMIC_PARTICLE_COUNT : newVal;
            needsGpuReinit = true; 
        }
    } else {
        this[paramName] = newVal;
    }

    if (needsGpuReinit) this._reinitGPUSimulation();
    else if (paramName === 'VISUAL_PARTICLE_SIZE') this._createParticleSystem();
    
    this._populateAdvancedControlsUI();
  }
  
  _populateAdvancedControlsUI() {
    const setCtrl = (id, val, exp, fix) => {
      const sE = this.uiElements[id + "Slider"], vE = this.uiElements[id + "Value"];
      if (sE && document.activeElement !== sE) sE.value = val;
      if (vE) {
        let dS = '';
        if (typeof val === 'number') {
          if (exp != null && (Math.abs(val) > 1e5 || (Math.abs(val) < 1e-4 && val !== 0))) dS = val.toExponential(exp);
          else if (fix != null) dS = val.toFixed(fix);
          else dS = val.toString();
        }
        vE.textContent = dS;
      }
    };
    
    setCtrl('advPlateThickness', this.PLATE_THICKNESS, null, 4);
    setCtrl('advPlateDensity', this.PLATE_DENSITY, null, 0);
    setCtrl('advEModulus', this.E_MODULUS, 2, null);
    setCtrl('advPoissonRatio', this.POISSON_RATIO, null, 2);
    setCtrl('advGridSize', this.GRID_SIZE, null, 0);
    setCtrl('advFDMSteps', this.FDM_STEPS_PER_FRAME, null, 0);
    setCtrl('advFDMDampingFactor', this.FDM_DAMPING_FACTOR, null, 6);
    setCtrl('advParticleCount', this.MAX_PARTICLE_COUNT, null, 0);
    setCtrl('advParticleForceBase', this.PARTICLE_FORCE_BASE, 1, null);
    setCtrl('advParticleDampingBase', this.PARTICLE_DAMPING_BASE, null, 3);
    setCtrl('advMaxParticleSpeed', this.MAX_PARTICLE_SPEED, null, 1);
    if(this.uiElements.advEnableRepulsion) this.uiElements.advEnableRepulsion.checked = this.ENABLE_PARTICLE_REPULSION;
    setCtrl('advRepulsionRadius', this.PARTICLE_REPULSION_RADIUS, null, 3);
    setCtrl('advRepulsionStrength', this.PARTICLE_REPULSION_STRENGTH, null, 4);
    setCtrl('advExcBaseAmp', this.EXCITATION_BASE_AMP, 1, null);
    setCtrl('advParticleSize', this.VISUAL_PARTICLE_SIZE, null, 3);
    setCtrl('advVisDeformScale', this.VISUAL_DEFORMATION_SCALE, null, 1);
    setCtrl('advMaxVisAmplitude', this.MAX_VISUAL_AMPLITUDE, null, 2);
  }

  _setupEventListeners() {
    window.addEventListener('resize', () => {
      this.camera.aspect = window.innerWidth / window.innerHeight;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(window.innerWidth, window.innerHeight);
      this.composer.setSize(window.innerWidth, window.innerHeight);
    }, false);

    window.addEventListener('keydown', (e) => this._handleKeyboardPiano(e));
    window.addEventListener('keyup', (e) => this._handleKeyboardPiano(e));
    
    const setupAdvControlListener = (baseId, varName) => {
      const slider = this.uiElements[baseId + 'Slider'];
      if (slider) {
        slider.addEventListener('input', () => this._applyAdvancedSettingChange(varName, parseFloat(slider.value)));
        slider.addEventListener('change', () => this._populateAdvancedControlsUI());
      }
    };
    
    const advControls = {
      'advPlateThickness': 'PLATE_THICKNESS', 'advPlateDensity': 'PLATE_DENSITY', 'advEModulus': 'E_MODULUS', 'advPoissonRatio': 'POISSON_RATIO',
      'advGridSize': 'GRID_SIZE', 'advFDMSteps': 'FDM_STEPS_PER_FRAME', 'advFDMDampingFactor': 'FDM_DAMPING_FACTOR',
      'advParticleCount': 'MAX_PARTICLE_COUNT', 'advParticleForceBase': 'PARTICLE_FORCE_BASE', 'advParticleDampingBase': 'PARTICLE_DAMPING_BASE',
      'advMaxParticleSpeed': 'MAX_PARTICLE_SPEED', 'advRepulsionRadius': 'PARTICLE_REPULSION_RADIUS', 'advRepulsionStrength': 'PARTICLE_REPULSION_STRENGTH',
      'advExcBaseAmp': 'EXCITATION_BASE_AMP', 'advParticleSize': 'VISUAL_PARTICLE_SIZE', 'advVisDeformScale': 'VISUAL_DEFORMATION_SCALE', 'advMaxVisAmplitude': 'MAX_VISUAL_AMPLITUDE'
    };
    for(const [id, variable] of Object.entries(advControls)) setupAdvControlListener(id, variable);
    
    if(this.uiElements.advEnableRepulsion) this.uiElements.advEnableRepulsion.addEventListener('change', e => this._applyAdvancedSettingChange('ENABLE_PARTICLE_REPULSION', e.target.checked));
    if(this.uiElements.resetAdvancedButton) this.uiElements.resetAdvancedButton.addEventListener('click', () => this._resetAllSettingsToDefaults(true));
    if(this.uiElements.resetSimulationButton) this.uiElements.resetSimulationButton.addEventListener('click', () => this._resetAllSettingsToDefaults(false));

    if(this.uiElements.frequencySlider) this.uiElements.frequencySlider.addEventListener('input', (e)=>{ const v = parseFloat(e.target.value); this.currentFrequency = 20 * Math.pow(20000 / 20, v / 100.0); this._setActiveDrivingMechanism('point'); });
    if(this.uiElements.setFrequencyButton) this.uiElements.setFrequencyButton.addEventListener('click', () => { this.currentFrequency = this._parseInputNumber(this.uiElements.frequencyInput.value, 273, false, 1, 999999); this._setActiveDrivingMechanism('point'); });
    if(this.uiElements.presetSelect) this.uiElements.presetSelect.addEventListener('change', (e) => { const val = e.target.value; if(val === "none") return; if(val === 'zvezda_lada'){ this.mParameter=4; this.nParameter=2; } else { const [m, n] = e.target.value.split(',').map(Number); this.mParameter = m; this.nParameter = n; } this._setActiveDrivingMechanism('modal'); this._updateModalParametersUI(); });
    
    const paramHandler = (param, value) => { this[param] = this._parseInputNumber(value, this[param], true, 0, 99); this._setActiveDrivingMechanism('modal'); this._updateModalParametersUI(); };
    if(this.uiElements.mParamSlider) this.uiElements.mParamSlider.addEventListener('input', e => paramHandler('mParameter', e.target.value));
    if(this.uiElements.nParamSlider) this.uiElements.nParamSlider.addEventListener('input', e => paramHandler('nParameter', e.target.value));

    if(this.uiElements.toggleFreezeButton) this.uiElements.toggleFreezeButton.addEventListener('click', () => { this.areParticlesFrozen = !this.areParticlesFrozen; this._updateUIToggleButtons(); });
    if(this.uiElements.toggleShadowsButton) this.uiElements.toggleShadowsButton.addEventListener('click', () => { this.enableShadows = !this.enableShadows; this._applyShadowSettings(); this._updateUIToggleButtons(); });
    if(this.uiElements.toggleStuckParticleCullingButton) this.uiElements.toggleStuckParticleCullingButton.addEventListener('click', () => { this.enableStuckParticleCulling = !this.enableStuckParticleCulling; this._updateUIToggleButtons(); });
    if(this.uiElements.toggleDynamicDensityButton) this.uiElements.toggleDynamicDensityButton.addEventListener('click', () => { this.enableDynamicParticleDensity = !this.enableDynamicParticleDensity; if(!this.enableDynamicParticleDensity) { this.ACTIVE_PARTICLE_COUNT = this.MAX_PARTICLE_COUNT; } this._updateUIToggleButtons(); });
    if (this.uiElements.toggleColorBySpeedButton) this.uiElements.toggleColorBySpeedButton.addEventListener('click', () => { this.enableColorBySpeed = !this.enableColorBySpeed; if (this.particleMeshMaterial) { this.particleMeshMaterial.uniforms.u_colorMode.value = this.enableColorBySpeed ? 1.0 : 0.0; } this._updateUIToggleButtons(); });
    if (this.uiElements.toggleSubtitlesButton) this.uiElements.toggleSubtitlesButton.addEventListener('click', () => { this.isSubtitlesEnabled = !this.isSubtitlesEnabled; this._updateUIToggleButtons(); });
  
    if(this.uiElements.audioFileInput) this.uiElements.audioFileInput.addEventListener('change', (e) => this._handleAudioFileSelection(e));
    if(this.uiElements.playUploadedAudioButton) this.uiElements.playUploadedAudioButton.addEventListener('click', () => { if (this.playlistFiles.length > 0) this._loadAndPlayTrack(this.currentPlaylistIndex === -1 ? 0 : this.currentPlaylistIndex); });
    if(this.uiElements.stopAudioButton) this.uiElements.stopAudioButton.addEventListener('click', () => this._stopLoadedAudioFilePlayback(false));
    if(this.uiElements.toggleAudioPauseButton) this.uiElements.toggleAudioPauseButton.addEventListener('click', () => this._togglePauseLoadedAudioFilePlayback());
    if(this.uiElements.nextTrackButton) this.uiElements.nextTrackButton.addEventListener('click', () => this._nextTrack());
    if(this.uiElements.prevTrackButton) this.uiElements.prevTrackButton.addEventListener('click', () => this._prevTrack());
    if(this.uiElements.toggleMicrophoneButton) this.uiElements.toggleMicrophoneButton.addEventListener('click', () => this._toggleMicrophoneInput());
    if(this.uiElements.toggleDesktopAudioButton) this.uiElements.toggleDesktopAudioButton.addEventListener('click', () => this._toggleDesktopAudio());
    if(this.uiElements.toggleSoundButton) this.uiElements.toggleSoundButton.addEventListener('click', () => this._toggleGeneratedSoundPlayback());
    if(this.uiElements.pianoOctaveSelect) this.uiElements.pianoOctaveSelect.addEventListener('change', e => this.currentPianoOctave = parseInt(e.target.value));
    
    if(this.uiElements.toggleLeftPanelButton) this.uiElements.toggleLeftPanelButton.addEventListener('click', () => this._togglePanel('controls', 'toggleLeftPanelButton', 'Скрыть левую панель', 'Показать левую панель', 'panel-hidden-left'));
    if(this.uiElements.toggleRightPanelButton) this.uiElements.toggleRightPanelButton.addEventListener('click', () => this._togglePanel('advanced-controls', 'toggleRightPanelButton', 'Скрыть правую панель', 'Показать правую панель', 'panel-hidden-right'));
  }
  
  _togglePanel(panelId, buttonId, hideText, showText, cssClass) {
    const panel = this.uiElements[panelId];
    const button = this.uiElements[buttonId];
    if (!panel || !button) return;
    
    const isHidden = panel.classList.toggle(cssClass);
    button.textContent = isHidden ? showText : hideText;
  }

  _applyShadowSettings() {
    this.renderer.shadowMap.enabled = this.enableShadows;
    this.dirLight1.castShadow = this.enableShadows;
    if (this.particlesMesh) this.particlesMesh.castShadow = this.enableShadows;
    if (this.groundPlane) this.groundPlane.visible = this.enableShadows;
    if (this.particleMeshMaterial) this.particleMeshMaterial.needsUpdate = true;
  }
  
  _updateFrequencyControlsUI() {
    if (!this.uiElements.freqValueText) return;
    this.uiElements.freqValueText.textContent = `${this.currentFrequency.toFixed(0)} Гц`;
    if (this.uiElements.frequencyInput && document.activeElement !== this.uiElements.frequencyInput) this.uiElements.frequencyInput.value = this.currentFrequency.toFixed(1);
    if (this.uiElements.frequencySlider && document.activeElement !== this.uiElements.frequencySlider) {
      const minLog = Math.log10(20); const maxLog = Math.log10(20000);
      this.uiElements.frequencySlider.value = 100 * (Math.log10(this.currentFrequency) - minLog) / (maxLog - minLog);
    }
  }

  _updateModalParametersUI() {
    if(this.uiElements.mParamValueText) this.uiElements.mParamValueText.textContent = this.mParameter;
    if(this.uiElements.mParamSlider) this.uiElements.mParamSlider.value = this.mParameter;
    if(this.uiElements.nParamValueText) this.uiElements.nParamValueText.textContent = this.nParameter;
    if(this.uiElements.nParamSlider) this.uiElements.nParamSlider.value = this.nParameter;
    if (this.uiElements.presetSelect) {
      const currentPresetVal = `${this.mParameter},${this.nParameter}`;
      const match = Array.from(this.uiElements.presetSelect.options).find(o => o.value === currentPresetVal);
      this.uiElements.presetSelect.value = match ? currentPresetVal : 'none';
      if (this.uiElements.presetSelect.value === 'none') {
          const opt = this.uiElements.presetSelect.options[0];
          opt.textContent = `Свои m, n (сейчас: ${this.mParameter},${this.nParameter})`;
      }
    }
  }

  _updateUIToggleButtons() {
    const setButtonState = (button, flag, onText, offText) => {
      if (!button) return;
      button.textContent = flag ? onText : offText;
      button.classList.toggle('button-on', flag);
      button.classList.toggle('button-off', !flag);
    };
    setButtonState(this.uiElements.toggleFreezeButton, this.areParticlesFrozen, "Частицы: Стоп", "Частицы: Движ.");
    setButtonState(this.uiElements.toggleShadowsButton, this.enableShadows, "Тени: Вкл", "Тени: Выкл");
    setButtonState(this.uiElements.toggleStuckParticleCullingButton, this.enableStuckParticleCulling, "Скрытие: Вкл", "Скрытие: Выкл");
    setButtonState(this.uiElements.toggleColorBySpeedButton, this.enableColorBySpeed, "Цвет (V): Вкл", "Цвет (V): Выкл");
    setButtonState(this.uiElements.toggleDynamicDensityButton, this.enableDynamicParticleDensity, "Динам. плотность: Вкл", "Динам. плотность: Выкл");
    setButtonState(this.uiElements.toggleSoundButton, this.isGeneratedSoundEnabled, "Звук: Вкл", "Звук: Выкл");
    setButtonState(this.uiElements.toggleSubtitlesButton, this.isSubtitlesEnabled, "Субтитры: Вкл", "Субтитры: Выкл");
    setButtonState(this.uiElements.toggleMicrophoneButton, this.isMicrophoneEnabled, "Микрофон: Вкл", "Микрофон: Выкл");
    setButtonState(this.uiElements.toggleDesktopAudioButton, this.isDesktopAudioEnabled, "Перехват звука: Вкл", "Перехват звука: Выкл");
  }
  
  _mapUIElements() {
    const ids = ['frequencySlider', 'freqValueText', 'frequencyInput', 'setFrequencyButton', 'particleSpeedSlider', 'speedValueText', 'presetSelect', 'mParamSlider', 'mParamValueText', 'nParamSlider', 'nParamValueText', 'toggleDesktopAudioButton', 'toggleSoundButton', 'toggleFreezeButton', 'toggleSubtitlesButton', 'toggleShadowsButton', 'toggleColorBySpeedButton', 'toggleStuckParticleCullingButton', 'toggleDynamicDensityButton', 'resetSimulationButton', 'activeModeDisplay', 'plateRotationSpeedSlider', 'plateRotationSpeedValue', 'stopRotationButton', 'audioFileInput', 'playUploadedAudioButton', 'stopAudioButton', 'toggleAudioPauseButton', 'audioInfoEl', 'lyricsInfoEl', 'nextTrackButton', 'prevTrackButton', 'toggleMicrophoneButton', 'microphoneInfoEl', 'audioProgressSlider', 'audioTimeDisplay', 'pitchDetectorInfo', 'pitch', 'note', 'detune_amt', 'detune', 'bpmInfo', 'bpmValue', 'bpmConfidence', 'pianoOctaveSelect', 'pianoContainer', 'pianoStatus', 'advPlateThicknessSlider', 'advPlateThicknessValue', 'advPlateDensitySlider', 'advPlateDensityValue', 'advEModulusSlider', 'advEModulusValue', 'advPoissonRatioSlider', 'advPoissonRatioValue', 'advGridSizeSlider', 'advGridSizeValue', 'advFDMStepsSlider', 'advFDMStepsValue', 'advFDMDampingFactorSlider', 'advFDMDampingFactorValue', 'advParticleCountSlider', 'advParticleCountValue', 'advParticleForceBaseSlider', 'advParticleForceBaseValue', 'advParticleDampingBaseSlider', 'advParticleDampingBaseValue', 'advMaxParticleSpeedSlider', 'advMaxParticleSpeedValue', 'advEnableRepulsion', 'advRepulsionRadiusSlider', 'advRepulsionRadiusValue', 'advRepulsionStrengthSlider', 'advRepulsionStrengthValue', 'advExcBaseAmpSlider', 'advExcBaseAmpValue', 'advParticleSizeSlider', 'advParticleSizeValue', 'advVisDeformScaleSlider', 'advVisDeformScaleValue', 'advMaxVisAmplitudeSlider', 'advMaxVisAmplitudeValue', 'resetAdvancedButton', 'controls', 'advanced-controls', 'toggleLeftPanelButton', 'toggleRightPanelButton', 'subtitle-container'];
    ids.forEach(id => {
      const el = document.getElementById(id);
      if (el) this.uiElements[id] = el;
    });
  }
  
  _storeDefaultSimulationSettings() {
    this.defaultAdvancedSettings = { PLATE_THICKNESS: PLATE_THICKNESS_DEFAULT, PLATE_DENSITY: PLATE_DENSITY_DEFAULT, E_MODULUS: E_MODULUS_DEFAULT, POISSON_RATIO: POISSON_RATIO_DEFAULT, GRID_SIZE: GPU_GRID_SIZE_DEFAULT, FDM_STEPS_PER_FRAME: GPU_FDM_STEPS_PER_FRAME_DEFAULT, FDM_DAMPING_FACTOR: 0.00005, MAX_PARTICLE_COUNT: GPU_PARTICLE_COUNT_DEFAULT, PARTICLE_FORCE_BASE: PARTICLE_FORCE_BASE_DEFAULT, PARTICLE_DAMPING_BASE: PARTICLE_DAMPING_BASE_DEFAULT, MAX_PARTICLE_SPEED: MAX_PARTICLE_SPEED_DEFAULT, VISUAL_PARTICLE_SIZE: VISUAL_PARTICLE_SIZE_DEFAULT, ENABLE_PARTICLE_REPULSION: ENABLE_REPULSION_DEFAULT, PARTICLE_REPULSION_RADIUS: REPULSION_RADIUS_DEFAULT, PARTICLE_REPULSION_STRENGTH: REPULSION_STRENGTH_DEFAULT, EXC_BASE_AMP: EXC_BASE_AMP_DEFAULT, VISUAL_DEFORMATION_SCALE: VISUAL_DEFORMATION_SCALE_DEFAULT, MAX_VISUAL_AMPLITUDE: MAX_VISUAL_AMPLITUDE_DEFAULT, };
  }
  
  _createPianoKeys() { 
    if (!this.uiElements.pianoContainer) return; 
    this.uiElements.pianoContainer.innerHTML = ''; 
    const keys = [{n:'C',t:'white'},{n:'C#',t:'black'},{n:'D',t:'white'},{n:'D#',t:'black'},{n:'E',t:'white'},{n:'F',t:'white'},{n:'F#',t:'black'},{n:'G',t:'white'},{n:'G#',t:'black'},{n:'A',t:'white'},{n:'A#',t:'black'},{n:'B',t:'white'}]; 
    let whiteOff=0; 
    keys.forEach(k=>{ 
        const div = document.createElement('div'); 
        div.className=`piano-key ${k.t}`; 
        div.dataset.note=k.n; 
        const span = document.createElement('span'); span.textContent = k.n; div.appendChild(span);
        if(k.t==='white'){ div.style.left=`${whiteOff*35}px`; whiteOff++; } 
        else { div.style.left=`${(whiteOff-1)*35 + 17.5}px`; }
        const press=(e)=>{ e.preventDefault(); this._handlePianoKeyPress(k.n, true, false, e.shiftKey); }; 
        const rel=(e)=>{ e.preventDefault(); this._handlePianoKeyPress(k.n, false, false, false); }; 
        div.addEventListener('mousedown',press); 
        div.addEventListener('mouseup',rel); 
        div.addEventListener('mouseleave',(e)=>{if(e.buttons===1&&this.activePianoKeys.has(k.n))this._handlePianoKeyPress(k.n,false, false, false);}); 
        div.addEventListener('touchstart',press,{passive:false}); 
        div.addEventListener('touchend',rel);
        this.uiElements.pianoContainer.appendChild(div); 
    }); 
  }

  _handleKeyboardPiano(event) {
    if (document.activeElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
    const note = this.keyToNoteMapping[event.code];
    if (note) {
      event.preventDefault();
      if (event.type === 'keydown' && !this.keyboardPressedKeys.has(event.code)) {
        this._handlePianoKeyPress(event.code, true, true, event.shiftKey);
      } else if (event.type === 'keyup' && this.keyboardPressedKeys.has(event.code)) {
        this._handlePianoKeyPress(event.code, false, true, false);
      }
    }
  }

  _handlePianoKeyPress(noteOrKeyCode, isPressed, isKeyboardEvent = false, isShiftPressed = false) {
    let noteName = isKeyboardEvent ? this.keyToNoteMapping[noteOrKeyCode] : noteOrKeyCode;
    if (!noteName) return;
    
    let actualOctave = this.currentPianoOctave;
    if (noteName.endsWith('5')) { actualOctave++; noteName = noteName.slice(0, -1); }
    if (isShiftPressed) actualOctave++;
    
    const keySet = isKeyboardEvent ? this.keyboardPressedKeys : this.activePianoKeys;
    const key = isKeyboardEvent ? noteOrKeyCode : noteName;
    
    if (isPressed) keySet.add(key); else keySet.delete(key);
    
    const pianoKeyElement = this.uiElements.pianoContainer?.querySelector(`.piano-key[data-note="${noteName}"]`);
    if(pianoKeyElement) pianoKeyElement.classList.toggle('active', isPressed);

    if (isPressed) {
      this.currentFrequency = 440 * Math.pow(2, (NOTE_TO_MIDI_NUMBER_OFFSET[noteName] + (actualOctave * 12) - 69) / 12);
      this._setActiveDrivingMechanism('point');
      if (this.isGeneratedSoundEnabled) { this._toggleGeneratedSoundPlayback(true); this._toggleGeneratedSoundPlayback(false); }
    }
  }

  _toggleGeneratedSoundPlayback(forceOff = false) {
    this.isGeneratedSoundEnabled = forceOff ? false : !this.isGeneratedSoundEnabled;

    if (this.generatedSoundOscillator) {
      this.generatedSoundOscillator.stop();
      this.generatedSoundOscillator = null;
    }
    if (this.isGeneratedSoundEnabled) {
      if (!this.mainAudioContext) this._setupWebAudioSystem();
      if (this.mainAudioContext.state === 'suspended') this.mainAudioContext.resume();
      this.generatedSoundOscillator = this.mainAudioContext.createOscillator();
      this.generatedSoundGainNode = this.mainAudioContext.createGain();
      this.generatedSoundOscillator.connect(this.generatedSoundGainNode).connect(this.mainAudioContext.destination);
      this.generatedSoundOscillator.type = 'sine';
      this.generatedSoundOscillator.frequency.setValueAtTime(this.actualAppliedFrequency, this.mainAudioContext.currentTime);
      this.generatedSoundGainNode.gain.setValueAtTime(0.15, this.mainAudioContext.currentTime);
      this.generatedSoundOscillator.start();
    }
    this._updateUIToggleButtons();
  }
  
  _createAudioElement() {
    const audio = document.createElement('audio');
    audio.crossOrigin = "anonymous";
    audio.addEventListener('loadedmetadata', () => {
        this.audioFileDuration = audio.duration;
    });
    audio.addEventListener('ended', () => this._nextTrack());
    return audio;
  }

  async _handleAudioFileSelection(event) {
    if (!event.target.files.length) return;
    await this._stopLoadedAudioFilePlayback(true);
    this.playlistFiles = Array.from(event.target.files);
    this.currentPlaylistIndex = -1;
    this._updateAudioFileUI();
  }
  
  async _loadAndPlayTrack(trackIndex) {
    if (trackIndex < 0 || trackIndex >= this.playlistFiles.length) {
      this._stopLoadedAudioFilePlayback(true);
      return;
    }
    this.currentPlaylistIndex = trackIndex;
    const file = this.playlistFiles[trackIndex];
    this.activeFetchID++;
    const currentFetchID = this.activeFetchID;
    
    this._findAndLoadLyrics(file, currentFetchID);
    this.audioElement.src = URL.createObjectURL(file);
    
    if (!this.audioFileSourceNode) {
        this.audioFileSourceNode = this.mainAudioContext.createMediaElementSource(this.audioElement);
        this.audioFileSourceNode.connect(this.mainAudioContext.destination);
        this.audioFileSourceNode.connect(this.pitchDetectorAnalyserNode);
        if(this.fftAnalyserNode) this.audioFileSourceNode.connect(this.fftAnalyserNode);
    }
      
    try {
        await this.audioElement.play();
        this._setActiveDrivingMechanism('audio');
        this.isAudioFilePlaying = true;
        this.isAudioFilePaused = false;
        this._updateAudioFileUI();
    } catch (e) {
      alert(`Error playing audio file: ${file.name}. ${e.message}`);
      this._nextTrack();
    }
  }

  _nextTrack() { this._loadAndPlayTrack((this.currentPlaylistIndex + 1) % this.playlistFiles.length); }
  _prevTrack() { this._loadAndPlayTrack((this.currentPlaylistIndex - 1 + this.playlistFiles.length) % this.playlistFiles.length); }

  _stopLoadedAudioFilePlayback(fullReset = false) {
    if (this.audioElement) {
        this.audioElement.pause();
        this.audioElement.removeAttribute('src'); 
        this.audioElement.load();
    }
    this.isAudioFilePlaying = false;
    this.isAudioFilePaused = false;
    if (this.drivingMechanism === 'audio') this._setActiveDrivingMechanism('modal');
    if (fullReset) {
      this.playlistFiles = [];
      this.currentPlaylistIndex = -1;
      if (this.uiElements.audioFileInput) this.uiElements.audioFileInput.value = "";
    }
    this._updateAudioFileUI();
  }
  
  _togglePauseLoadedAudioFilePlayback() {
    if (!this.audioElement || !this.isAudioFilePlaying) return;
    if (this.audioElement.paused) {
      this.audioElement.play();
      this.isAudioFilePaused = false;
    } else {
      this.audioElement.pause();
      this.isAudioFilePaused = true;
    }
    if (this.uiElements.toggleAudioPauseButton) this.uiElements.toggleAudioPauseButton.textContent = this.isAudioFilePaused ? "Продолжить" : "Пауза";
  }

  _updateAudioFileUI() {
    const hasFiles = this.playlistFiles.length > 0;
    const isPlayingOrPaused = this.isAudioFilePlaying || this.isAudioFilePaused;
    
    const ui = this.uiElements;
    if (ui.playUploadedAudioButton) ui.playUploadedAudioButton.style.display = hasFiles && !isPlayingOrPaused ? 'block' : 'none';
    if (ui.stopAudioButton) ui.stopAudioButton.style.display = isPlayingOrPaused ? 'block' : 'none';
    if (ui.toggleAudioPauseButton) ui.toggleAudioPauseButton.style.display = isPlayingOrPaused ? 'block' : 'none';
    if (ui.nextTrackButton) ui.nextTrackButton.style.display = hasFiles && this.playlistFiles.length > 1 ? 'block' : 'none';
    if (ui.prevTrackButton) ui.prevTrackButton.style.display = hasFiles && this.playlistFiles.length > 1 ? 'block' : 'none';
    if (ui.audioProgressSlider) ui.audioProgressSlider.style.display = isPlayingOrPaused ? 'block' : 'none';
    if (ui.audioTimeDisplay) ui.audioTimeDisplay.style.display = isPlayingOrPaused ? 'block' : 'none';
    
    if (ui.audioInfoEl) {
        if(isPlayingOrPaused) ui.audioInfoEl.textContent = `Трек: ${this.playlistFiles[this.currentPlaylistIndex].name}`;
        else ui.audioInfoEl.textContent = hasFiles ? `${this.playlistFiles.length} трек(ов) в плейлисте` : 'Аудио не загружено';
    }
  }

  _updateAudioFileProgressControlsUI() {
    if (!this.audioElement || !isFinite(this.audioFileDuration) || this.audioFileDuration === 0) return;
    const currentTime = this.audioElement.currentTime;
    if (this.uiElements.audioProgressSlider && document.activeElement !== this.uiElements.audioProgressSlider) {
      this.uiElements.audioProgressSlider.value = (currentTime / this.audioFileDuration) * 100;
    }
    if (this.uiElements.audioTimeDisplay) {
      this.uiElements.audioTimeDisplay.textContent = `${this._formatTimeMMSS(currentTime)} / ${this._formatTimeMMSS(this.audioFileDuration)}`;
    }
  }

  async _toggleMicrophoneInput() {
    if (this.isMicrophoneEnabled) {
      if(this.microphoneStream) this.microphoneStream.getTracks().forEach(track => track.stop());
      if(this.microphoneSourceNode) this.microphoneSourceNode.disconnect();
      this.isMicrophoneEnabled = false;
      if(this.drivingMechanism === 'microphone') this._setActiveDrivingMechanism('modal');
    } else {
      try {
        await this._stopLoadedAudioFilePlayback(true);
        this.microphoneStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.microphoneSourceNode = this.mainAudioContext.createMediaStreamSource(this.microphoneStream);
        this.microphoneSourceNode.connect(this.pitchDetectorAnalyserNode);
        if(this.fftAnalyserNode) this.microphoneSourceNode.connect(this.fftAnalyserNode);
        this.isMicrophoneEnabled = true;
        this._setActiveDrivingMechanism('microphone');
      } catch (e) { alert('Доступ к микрофону запрещен или невозможен.'); }
    }
    this._updateUIToggleButtons();
  }

  async _toggleDesktopAudio() {
    if (this.isDesktopAudioEnabled) {
      if(this.desktopStream) this.desktopStream.getTracks().forEach(track => track.stop());
      if(this.desktopAudioSourceNode) this.desktopAudioSourceNode.disconnect();
      this.isDesktopAudioEnabled = false;
      if(this.drivingMechanism === 'desktop_audio') this._setActiveDrivingMechanism('modal');
    } else {
      try {
        if (!navigator.mediaDevices.getDisplayMedia) throw new Error("getDisplayMedia not supported.");
        await this._stopLoadedAudioFilePlayback(true);
        this.desktopStream = await navigator.mediaDevices.getDisplayMedia({ audio: true, video: true });
        if (this.desktopStream.getAudioTracks().length === 0) {
          this.desktopStream.getTracks().forEach(track => track.stop());
          alert('Аудио не было предоставлено. Пожалуйста, убедитесь, что вы поставили галочку "Поделиться звуком системы".');
          return;
        }
        this.desktopAudioSourceNode = this.mainAudioContext.createMediaStreamSource(this.desktopStream);
        this.desktopAudioSourceNode.connect(this.pitchDetectorAnalyserNode);
        if(this.fftAnalyserNode) this.desktopAudioSourceNode.connect(this.fftAnalyserNode);
        this.isDesktopAudioEnabled = true;
        this._setActiveDrivingMechanism('desktop_audio');
      } catch (e) { alert(`Не удалось захватить аудио с экрана: ${e.message}`); }
    }
    this._updateUIToggleButtons();
  }
  
  _findAndLoadLyrics(file, fetchID) {
    if (typeof jsmediatags === 'undefined') {
      if(this.uiElements.lyricsInfoEl) this.uiElements.lyricsInfoEl.textContent = 'Библиотека для субтитров не найдена.';
      return;
    }
    jsmediatags.read(file, {
      onSuccess: (tag) => {
        if (fetchID !== this.activeFetchID) return;
        const lyricsData = tag.tags.lyrics || tag.tags.USLT;
        if (lyricsData) {
          this.currentSubtitles = this._parseLRC(typeof lyricsData === 'string' ? lyricsData : lyricsData.lyrics);
          if(this.uiElements.lyricsInfoEl) this.uiElements.lyricsInfoEl.textContent = this.currentSubtitles.length > 0 ? 'Субтитры найдены.' : 'Субтитры не найдены.';
        } else {
          if(this.uiElements.lyricsInfoEl) this.uiElements.lyricsInfoEl.textContent = 'Субтитры не найдены.';
        }
      },
      onError: () => {
        if(this.uiElements.lyricsInfoEl) this.uiElements.lyricsInfoEl.textContent = 'Ошибка чтения тегов.';
      }
    });
  }

  _parseLRC(lrcText) {
    const lines = lrcText.split('\n');
    const subtitles = [];
    lines.forEach(line => {
      const match = line.match(/\[(\d{2}):(\d{2})[.:](\d{2,3})\]/);
      if (match) {
        const time = parseInt(match[1]) * 60 + parseInt(match[2]) + parseInt(match[3].padEnd(3, '0')) / 1000;
        const text = line.replace(/\[.*?\]/g, '').trim();
        if (text) subtitles.push({ time, text });
      }
    });
    return subtitles.sort((a,b) => a.time - b.time);
  }

  _updateSubtitles() {
    const subContainer = this.uiElements.subtitleContainer;
    if (!this.isSubtitlesEnabled || !subContainer) return;
    
    if (!this.audioElement || this.currentSubtitles.length === 0) {
      if (subContainer.textContent !== '') {
        subContainer.textContent = '';
        subContainer.classList.remove('visible');
      }
      return;
    }
    let low = 0, high = this.currentSubtitles.length - 1, bestIndex = -1;
    while(low <= high) {
      const mid = Math.floor((low + high) / 2);
      if (this.currentSubtitles[mid].time <= this.audioElement.currentTime) {
        bestIndex = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    const newText = (bestIndex !== -1) ? this.currentSubtitles[bestIndex].text : '';
    if (subContainer.textContent !== newText) {
      subContainer.textContent = newText;
      subContainer.classList.toggle('visible', newText !== '');
    }
  }

  _autocorrelationForBPM(data) {
    const N = data.length;
    const mean = data.reduce((a, b) => a + b, 0) / N;
    const variance = data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / N;
    if (variance === 0) return new Array(N).fill(0);
    const ac = new Array(N).fill(0);
    for (let lag = 0; lag < N; lag++) {
      let sum = 0;
      for (let i = 0; i < N - lag; i++) sum += (data[i] - mean) * (data[i + lag] - mean);
      ac[lag] = sum / ((N - lag) * variance);
    }
    return ac;
  }

  _updateBPM() {
    if (this.onsetHistory.length < 50) return;
    const ac = this._autocorrelationForBPM(this.onsetHistory);
    let maxAc = -1;
    let bestLag = -1;
    
    const minLag = Math.floor(BPM_ANALYSIS_SAMPLE_RATE * 60 / 220); 
    const maxLag = Math.floor(BPM_ANALYSIS_SAMPLE_RATE * 60 / 60);

    for (let lag = minLag; lag <= maxLag && lag < ac.length; lag++) {
      if (ac[lag] > maxAc) {
        maxAc = ac[lag];
        bestLag = lag;
      }
    }
    if (bestLag !== -1 && maxAc > 0.1) {
      const bpm = (60 * BPM_ANALYSIS_SAMPLE_RATE) / bestLag;
      const smoothedBPM = this.lastBPM ? 0.9 * this.lastBPM + 0.1 * bpm : bpm;
      this.lastBPM = smoothedBPM;
      if (this.uiElements.bpmValue) this.uiElements.bpmValue.textContent = Math.round(smoothedBPM);
      if (this.uiElements.bpmConfidence) this.uiElements.bpmConfidence.textContent = `${Math.min(100, maxAc * 150).toFixed(0)}%`;
      if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'block';
    }
  }
}

async function main() {
  try {
    const response = await fetch('./data/bessel_roots.json');
    if (!response.ok) throw new Error(`Failed to load essential data 'bessel_roots.json': ${response.statusText}`);
    const besselRootsData = await response.json();
    new ChladniSimulator(besselRootsData);
  } catch (error) {
    console.error("Critical boot error:", error);
    document.body.innerHTML = `<div style="color: #e06c75; background-color:#282c34; padding: 20px; text-align: center; font-family: sans-serif;"><h1>Критическая ошибка</h1><p>Не удалось загрузить необходимые для работы данные. Проверьте консоль.</p><p style="margin-top: 10px; color: #abb2bf;">${error.message}</p></div>`;
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', main);
} else {
  main();
}
