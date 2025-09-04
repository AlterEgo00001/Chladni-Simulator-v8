class GPUComputeManagerOptimized {
    constructor(renderer) {
        this.renderer = renderer;
        this.gl = null;
        
        // Compute shaders
        this.fdmComputeShader = null;
        this.initFieldComputeShader = null;
        this.particleComputeShader = null;
        this.initParticlesComputeShader = null;
        
        // Buffers с оптимизированным выравниванием
        this.currentBuffer = null;
        this.previousBuffer = null;
        this.nextBuffer = null;
        this.excitationBuffer = null;
        this.particlePositionsBuffer = null;
        this.particleVelocitiesBuffer = null;
        this.particleStatesBuffer = null;
        this.outputPositionsBuffer = null;
        this.outputVelocitiesBuffer = null;
        this.outputStatesBuffer = null;
        
        // Uniform buffers с оптимизированным выравниванием
        this.fdmUniforms = null;
        this.initUniforms = null;
        this.particleUniforms = null;
        this.initParticleUniforms = null;
        
        // Compute programs
        this.fdmProgram = null;
        this.initFieldProgram = null;
        this.particleProgram = null;
        this.initParticlesProgram = null;
        
        // Оптимизированные параметры
        this.workGroupSizeX = 32;
        this.workGroupSizeY = 32;
        this.particleWorkGroupSize = 512;
        
        this.initialized = false;
        this.performanceMode = 'balanced'; // 'balanced', 'performance', 'quality'
    }
    
    async initialize() {
        try {
            // Получение WebGL2 контекста
            this.gl = this.renderer.getContext();
            
            // Проверка поддержки WebGL2
            if (!this.gl) {
                throw new Error('WebGL2 context not available');
            }
            
            // Проверка поддержки compute shaders
            const computeShaderExtension = this.gl.getExtension('WEBGL_compute_shader');
            if (!computeShaderExtension) {
                throw new Error('Compute shader extension not supported');
            }
            
            // Проверка поддержки SSBO
            const ssboExtension = this.gl.getExtension('WEBGL_shader_storage_buffer_object');
            if (!ssboExtension) {
                throw new Error('Shader storage buffer object extension not supported');
            }
            
            // Определение оптимального размера рабочей группы
            this.optimizeWorkGroupSizes();
            
            // Загрузка оптимизированных compute shaders
            await this.loadOptimizedComputeShaders();
            
            // Создание программ
            this.createComputePrograms();
            
            // Создание буферов с оптимизированным выравниванием
            this.createOptimizedBuffers();
            
            this.initialized = true;
            console.log('GPU Compute Manager (Optimized) initialized successfully');
        } catch (error) {
            console.log('GPU Compute Manager: Compute shader extension not supported, falling back to CPU');
            this.initialized = false;
            throw error;
        }
    }
    
    optimizeWorkGroupSizes() {
        // Адаптивная оптимизация размера рабочей группы
        const maxWorkGroupSize = this.gl.getParameter(this.gl.MAX_COMPUTE_WORK_GROUP_SIZE_X);
        
        if (maxWorkGroupSize >= 1024) {
            this.particleWorkGroupSize = 1024;
        } else if (maxWorkGroupSize >= 512) {
            this.particleWorkGroupSize = 512;
        } else {
            this.particleWorkGroupSize = 256;
        }
        
        // Оптимизация для FDM
        if (maxWorkGroupSize >= 64) {
            this.workGroupSizeX = 32;
            this.workGroupSizeY = 32;
        } else {
            this.workGroupSizeX = 16;
            this.workGroupSizeY = 16;
        }
    }
    
    async loadOptimizedComputeShaders() {
        // Загрузка оптимизированных шейдеров
        const fdmShaderSource = await this.loadShaderFile('shaders/fdm_compute_optimized.glsl');
        const initFieldShaderSource = await this.loadShaderFile('shaders/init_field_compute.glsl');
        const particleShaderSource = await this.loadShaderFile('shaders/particle_compute_optimized.glsl');
        const initParticlesShaderSource = await this.loadShaderFile('shaders/init_particles_compute.glsl');
        
        // Компиляция шейдеров с оптимизациями
        this.fdmComputeShader = this.compileOptimizedComputeShader(fdmShaderSource);
        this.initFieldComputeShader = this.compileOptimizedComputeShader(initFieldShaderSource);
        this.particleComputeShader = this.compileOptimizedComputeShader(particleShaderSource);
        this.initParticlesComputeShader = this.compileOptimizedComputeShader(initParticlesShaderSource);
    }
    
    async loadShaderFile(filename) {
        try {
            const response = await fetch(filename);
            if (!response.ok) {
                // Fallback к оригинальным шейдерам
                const fallbackResponse = await fetch(filename.replace('_optimized', ''));
                if (!fallbackResponse.ok) {
                    throw new Error(`Failed to load shader file: ${filename}`);
                }
                return await fallbackResponse.text();
            }
            return await response.text();
        } catch (error) {
            console.error(`Error loading shader file ${filename}:`, error);
            throw error;
        }
    }
    
    compileOptimizedComputeShader(source) {
        if (!this.gl) {
            throw new Error('WebGL2 context not initialized');
        }
        
        const shader = this.gl.createShader(this.gl.COMPUTE_SHADER);
        if (!shader) {
            throw new Error('Failed to create compute shader');
        }
        
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);
        
        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            const error = this.gl.getShaderInfoLog(shader);
            this.gl.deleteShader(shader);
            throw new Error(`Compute shader compilation failed: ${error}`);
        }
        
        return shader;
    }
    
    createComputePrograms() {
        if (!this.gl) {
            throw new Error('WebGL2 context not initialized');
        }
        
        // Создание программ для compute shaders
        this.fdmProgram = this.gl.createProgram();
        if (!this.fdmProgram) {
            throw new Error('Failed to create FDM program');
        }
        this.gl.attachShader(this.fdmProgram, this.fdmComputeShader);
        this.gl.linkProgram(this.fdmProgram);
        
        this.initFieldProgram = this.gl.createProgram();
        if (!this.initFieldProgram) {
            throw new Error('Failed to create init field program');
        }
        this.gl.attachShader(this.initFieldProgram, this.initFieldComputeShader);
        this.gl.linkProgram(this.initFieldProgram);
        
        this.particleProgram = this.gl.createProgram();
        if (!this.particleProgram) {
            throw new Error('Failed to create particle program');
        }
        this.gl.attachShader(this.particleProgram, this.particleComputeShader);
        this.gl.linkProgram(this.particleProgram);
        
        this.initParticlesProgram = this.gl.createProgram();
        if (!this.initParticlesProgram) {
            throw new Error('Failed to create init particles program');
        }
        this.gl.attachShader(this.initParticlesProgram, this.initParticlesComputeShader);
        this.gl.linkProgram(this.initParticlesProgram);
        
        // Проверка линковки
        [this.fdmProgram, this.initFieldProgram, this.particleProgram, this.initParticlesProgram].forEach(program => {
            if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
                const error = this.gl.getProgramInfoLog(program);
                throw new Error(`Program linking failed: ${error}`);
            }
        });
    }
    
    createOptimizedBuffers() {
        if (!this.gl) {
            throw new Error('WebGL2 context not initialized');
        }
        
        // Создание uniform буферов с оптимизированным выравниванием
        this.fdmUniforms = this.gl.createBuffer();
        this.initUniforms = this.gl.createBuffer();
        this.particleUniforms = this.gl.createBuffer();
        this.initParticleUniforms = this.gl.createBuffer();
        
        // Проверка создания буферов
        if (!this.fdmUniforms || !this.initUniforms || !this.particleUniforms || !this.initParticleUniforms) {
            throw new Error('Failed to create uniform buffers');
        }
    }
    
    createFDMBuffers(gridSize) {
        if (!this.gl) {
            throw new Error('WebGL2 context not initialized');
        }
        
        const bufferSize = gridSize * gridSize * 4; // float32 с оптимизированным выравниванием
        
        // Создание буферов для FDM с оптимизированным выравниванием
        this.currentBuffer = this.gl.createBuffer();
        if (!this.currentBuffer) {
            throw new Error('Failed to create current buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.currentBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, bufferSize, this.gl.DYNAMIC_DRAW);
        
        this.previousBuffer = this.gl.createBuffer();
        if (!this.previousBuffer) {
            throw new Error('Failed to create previous buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.previousBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, bufferSize, this.gl.DYNAMIC_DRAW);
        
        this.nextBuffer = this.gl.createBuffer();
        if (!this.nextBuffer) {
            throw new Error('Failed to create next buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.nextBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, bufferSize, this.gl.DYNAMIC_DRAW);
        
        this.excitationBuffer = this.gl.createBuffer();
        if (!this.excitationBuffer) {
            throw new Error('Failed to create excitation buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.excitationBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, bufferSize, this.gl.DYNAMIC_DRAW);
    }
    
    createParticleBuffers(particleCount) {
        if (!this.gl) {
            throw new Error('WebGL2 context not initialized');
        }
        
        const positionSize = particleCount * 16; // vec4 с оптимизированным выравниванием
        const velocitySize = particleCount * 16; // vec4 с оптимизированным выравниванием
        const stateSize = particleCount * 4; // int с оптимизированным выравниванием
        
        // Создание буферов для частиц с оптимизированным выравниванием
        this.particlePositionsBuffer = this.gl.createBuffer();
        if (!this.particlePositionsBuffer) {
            throw new Error('Failed to create particle positions buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.particlePositionsBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, positionSize, this.gl.DYNAMIC_DRAW);
        
        this.particleVelocitiesBuffer = this.gl.createBuffer();
        if (!this.particleVelocitiesBuffer) {
            throw new Error('Failed to create particle velocities buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.particleVelocitiesBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, velocitySize, this.gl.DYNAMIC_DRAW);
        
        this.particleStatesBuffer = this.gl.createBuffer();
        if (!this.particleStatesBuffer) {
            throw new Error('Failed to create particle states buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.particleStatesBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, stateSize, this.gl.DYNAMIC_DRAW);
        
        // Выходные буферы для частиц
        this.outputPositionsBuffer = this.gl.createBuffer();
        if (!this.outputPositionsBuffer) {
            throw new Error('Failed to create output positions buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.outputPositionsBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, positionSize, this.gl.DYNAMIC_DRAW);
        
        this.outputVelocitiesBuffer = this.gl.createBuffer();
        if (!this.outputVelocitiesBuffer) {
            throw new Error('Failed to create output velocities buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.outputVelocitiesBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, velocitySize, this.gl.DYNAMIC_DRAW);
        
        this.outputStatesBuffer = this.gl.createBuffer();
        if (!this.outputStatesBuffer) {
            throw new Error('Failed to create output states buffer');
        }
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.outputStatesBuffer);
        this.gl.bufferData(this.gl.SHADER_STORAGE_BUFFER, stateSize, this.gl.DYNAMIC_DRAW);
    }
    
    updateFDMUniforms(uniforms) {
        if (!this.initialized || !this.gl || !this.fdmUniforms) {
            return;
        }
        this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.fdmUniforms);
        this.gl.bufferData(this.gl.UNIFORM_BUFFER, new Float32Array(uniforms), this.gl.DYNAMIC_DRAW);
    }
    
    updateInitUniforms(uniforms) {
        if (!this.initialized || !this.gl || !this.initUniforms) {
            return;
        }
        this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.initUniforms);
        this.gl.bufferData(this.gl.UNIFORM_BUFFER, new Float32Array(uniforms), this.gl.DYNAMIC_DRAW);
    }
    
    updateParticleUniforms(uniforms) {
        if (!this.initialized || !this.gl || !this.particleUniforms) {
            return;
        }
        this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.particleUniforms);
        this.gl.bufferData(this.gl.UNIFORM_BUFFER, new Float32Array(uniforms), this.gl.DYNAMIC_DRAW);
    }
    
    updateInitParticleUniforms(uniforms) {
        if (!this.initialized || !this.gl || !this.initParticleUniforms) {
            return;
        }
        this.gl.bindBuffer(this.gl.UNIFORM_BUFFER, this.initParticleUniforms);
        this.gl.bufferData(this.gl.UNIFORM_BUFFER, new Float32Array(uniforms), this.gl.DYNAMIC_DRAW);
    }
    
    runFDMCompute(gridSize, numSteps) {
        if (!this.initialized || !this.gl || !this.fdmProgram) return;
        
        this.gl.useProgram(this.fdmProgram);
        
        // Привязка буферов
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 0, this.currentBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 1, this.previousBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 2, this.nextBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 3, this.excitationBuffer);
        this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.fdmUniforms);
        
        // Оптимизированный запуск compute shader
        const workGroupsX = Math.ceil(gridSize / this.workGroupSizeX);
        const workGroupsY = Math.ceil(gridSize / this.workGroupSizeY);
        
        for (let step = 0; step < numSteps; step++) {
            this.gl.dispatchCompute(workGroupsX, workGroupsY, 1);
            this.gl.memoryBarrier(this.gl.SHADER_STORAGE_BARRIER_BIT);
            
            // Обмен буферами
            [this.previousBuffer, this.currentBuffer, this.nextBuffer] = 
            [this.currentBuffer, this.nextBuffer, this.previousBuffer];
        }
    }
    
    runInitFieldCompute(gridSize) {
        if (!this.initialized || !this.gl || !this.initFieldProgram) return;
        
        this.gl.useProgram(this.initFieldProgram);
        
        // Привязка буферов
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 0, this.currentBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 1, this.previousBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 2, this.excitationBuffer);
        this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.initUniforms);
        
        // Оптимизированный запуск compute shader
        const workGroupsX = Math.ceil(gridSize / this.workGroupSizeX);
        const workGroupsY = Math.ceil(gridSize / this.workGroupSizeY);
        
        this.gl.dispatchCompute(workGroupsX, workGroupsY, 1);
        this.gl.memoryBarrier(this.gl.SHADER_STORAGE_BARRIER_BIT);
    }
    
    runParticleCompute(particleCount) {
        if (!this.initialized || !this.gl || !this.particleProgram) return;
        
        this.gl.useProgram(this.particleProgram);
        
        // Привязка буферов
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 0, this.currentBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 1, this.particlePositionsBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 2, this.particleVelocitiesBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 3, this.particleStatesBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 4, this.outputPositionsBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 5, this.outputVelocitiesBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 6, this.outputStatesBuffer);
        this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.particleUniforms);
        
        // Оптимизированный запуск compute shader
        const workGroupsX = Math.ceil(particleCount / this.particleWorkGroupSize);
        
        this.gl.dispatchCompute(workGroupsX, 1, 1);
        this.gl.memoryBarrier(this.gl.SHADER_STORAGE_BARRIER_BIT);
        
        // Обмен буферами
        [this.particlePositionsBuffer, this.outputPositionsBuffer] = 
        [this.outputPositionsBuffer, this.particlePositionsBuffer];
        
        [this.particleVelocitiesBuffer, this.outputVelocitiesBuffer] = 
        [this.outputVelocitiesBuffer, this.particleVelocitiesBuffer];
        
        [this.particleStatesBuffer, this.outputStatesBuffer] = 
        [this.outputStatesBuffer, this.particleStatesBuffer];
    }
    
    runInitParticlesCompute(particleCount) {
        if (!this.initialized || !this.gl || !this.initParticlesProgram) return;
        
        this.gl.useProgram(this.initParticlesProgram);
        
        // Привязка буферов
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 0, this.particlePositionsBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 1, this.particleVelocitiesBuffer);
        this.gl.bindBufferBase(this.gl.SHADER_STORAGE_BUFFER, 2, this.particleStatesBuffer);
        this.gl.bindBufferBase(this.gl.UNIFORM_BUFFER, 0, this.initParticleUniforms);
        
        // Оптимизированный запуск compute shader
        const workGroupsX = Math.ceil(particleCount / this.particleWorkGroupSize);
        
        this.gl.dispatchCompute(workGroupsX, 1, 1);
        this.gl.memoryBarrier(this.gl.SHADER_STORAGE_BARRIER_BIT);
    }
    
    getFDMData(gridSize) {
        if (!this.initialized || !this.gl || !this.currentBuffer) {
            return new Float32Array(gridSize * gridSize);
        }
        const data = new Float32Array(gridSize * gridSize);
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.currentBuffer);
        this.gl.getBufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, data);
        return data;
    }
    
    getParticleData(particleCount) {
        if (!this.initialized || !this.gl || !this.particlePositionsBuffer || !this.particleVelocitiesBuffer || !this.particleStatesBuffer) {
            return { 
                positions: new Float32Array(particleCount * 4),
                velocities: new Float32Array(particleCount * 4),
                states: new Int32Array(particleCount)
            };
        }
        
        const positions = new Float32Array(particleCount * 4);
        const velocities = new Float32Array(particleCount * 4);
        const states = new Int32Array(particleCount);
        
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.particlePositionsBuffer);
        this.gl.getBufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, positions);
        
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.particleVelocitiesBuffer);
        this.gl.getBufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, velocities);
        
        this.gl.bindBuffer(this.gl.SHADER_STORAGE_BUFFER, this.particleStatesBuffer);
        this.gl.getBufferSubData(this.gl.SHADER_STORAGE_BUFFER, 0, states);
        
        return { positions, velocities, states };
    }
    
    setPerformanceMode(mode) {
        this.performanceMode = mode;
        
        switch (mode) {
            case 'performance':
                this.workGroupSizeX = 64;
                this.workGroupSizeY = 64;
                this.particleWorkGroupSize = 1024;
                break;
            case 'balanced':
                this.workGroupSizeX = 32;
                this.workGroupSizeY = 32;
                this.particleWorkGroupSize = 512;
                break;
            case 'quality':
                this.workGroupSizeX = 16;
                this.workGroupSizeY = 16;
                this.particleWorkGroupSize = 256;
                break;
        }
    }
    
    dispose() {
        if (!this.gl) return;
        
        // Очистка ресурсов
        if (this.fdmComputeShader) this.gl.deleteShader(this.fdmComputeShader);
        if (this.initFieldComputeShader) this.gl.deleteShader(this.initFieldComputeShader);
        if (this.particleComputeShader) this.gl.deleteShader(this.particleComputeShader);
        if (this.initParticlesComputeShader) this.gl.deleteShader(this.initParticlesComputeShader);
        
        if (this.fdmProgram) this.gl.deleteProgram(this.fdmProgram);
        if (this.initFieldProgram) this.gl.deleteProgram(this.initFieldProgram);
        if (this.particleProgram) this.gl.deleteProgram(this.particleProgram);
        if (this.initParticlesProgram) this.gl.deleteProgram(this.initParticlesProgram);
        
        // Удаление буферов
        [this.currentBuffer, this.previousBuffer, this.nextBuffer, this.excitationBuffer,
         this.particlePositionsBuffer, this.particleVelocitiesBuffer, this.particleStatesBuffer,
         this.outputPositionsBuffer, this.outputVelocitiesBuffer, this.outputStatesBuffer,
         this.fdmUniforms, this.initUniforms, this.particleUniforms, this.initParticleUniforms].forEach(buffer => {
            if (buffer) this.gl.deleteBuffer(buffer);
        });
        
        this.initialized = false;
    }
}
