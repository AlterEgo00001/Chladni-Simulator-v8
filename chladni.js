
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const LANG_PACK = {
    'ru': {
        'welcome_title': 'Симулятор Хладни v7.3.55',
        'welcome_body': `
            <p>Я покакал. Это интерактивный симулятор, основанный на практических изучениях Хладни и других мазафакеров в виде корней Бесселя и других чуваков.</p>
            <p>Проект поставляется по лицензии <b>"АЩЕ ПОХУЙ"</b>, т.е. бесплатная, поскольку это сборная солянка из всех практических работ в интернете. Моя задача была просто свести всю хуйню воедино, что, по сути, и получилось благодаря двум ИИ — Gemini и Grok. Проект мог бы быть масштабнее, но пока у меня ограничения в 1 000 000 токенов, но и этого было достаточно.</p>
            <p>Практических применений много, т.к. проект является уникальной интерактивной мазафакой с точки зрения Физики и других Алгебраических штук.</p>
            <p>Отдельно спасибо: Даниилу Дмитриевичу =)</p>
            <p>Для связи со мной <b>https://t.me/unknown_sector</b></p>
            <p><b>Что я не успел сделать, но сделаю:</b><br>
            - Перенести на Github и развернуть на Github Pages. (Выполнено) <br>
            - Перенести на Electron для реализации проекта с использованием GPU (Win/Linux).<br>
            - Интегрировать возможность работы Client-Server (сейчас всё выполняется у клиента, и это нагружает ресурсы пользователя).</p>
            <p>В будущем, когда будет время, буду дорабатывать проект, а пока я прячусь от налоговой. Но донат никто не отменял, поэтому вот адрес моего кошелька TON в телеге: <b>UQAY5_p2plWbshvrITsPk5TJ4CKBneJImdVUV-8-MBOA4Lhh</b></p>
            <div class="welcome-image-container">
                <img src="./img/111.png" alt="Chladni Pattern Example" border="0" />
                <img src="./img/222.jpg" alt="Project Logo" border="0" />
            </div>
        `,
        'lang_toggle': 'Switch to English',
        'show_prompt': 'Промпт Проекта',
        'close_welcome': 'Начать',
        'prompt_title': 'Промпт / Техническое Задание Проекта',
        'close_modal': 'Закрыть',
        'toggle_left_panel_show': 'Скрыть левую панель',
        'toggle_left_panel_hide': 'Показать левую панель',
        'toggle_right_panel_show': 'Скрыть правую панель',
        'toggle_right_panel_hide': 'Показать правую панель',
        'btn_show_welcome': 'Помощь / Welcome',
        'info_controls': 'Удерживайте ЛКМ и двигайте для вращения камеры.<br>Используйте Колесо Мыши для приближения/отдаления.',
        'info_version': 'Симулятор v7.3.55 (Интерактивное Обучение).',
        'legend_main_params': 'Основные параметры',
        'label_freq_slider': 'Частота (ползунок):',
        'label_freq_input': 'Частота (ввод Гц):',
        'btn_set': 'Установить',
        'label_sim_speed': 'Скорость симуляции:',
        'label_presets': 'Предустановки (m,n для круга):',
        'preset_custom': 'Свои m, n (сейчас: {m},{n})',
        'preset_custom_inactive': 'Свои m, n (неактивно)',
        'preset_lada': 'Звезда Лады (4,2)',
        'param_m': 'Мода m:',
        'btn_set_m': 'Уст. m',
        'param_n': 'Мода n:',
        'btn_set_n': 'Уст. n',
        'legend_sim_controls': 'Управление симуляцией',
        'btn_desktop_audio_on': 'Перехват звука: Вкл', 'btn_desktop_audio_off': 'Перехват звука: Выкл',
        'btn_sound_on': 'Звук: Вкл', 'btn_sound_off': 'Звук: Выкл',
        'btn_freeze_on': 'Частицы: Замороз.', 'btn_freeze_off': 'Частицы: Движ.',
        'btn_subs_on': 'Субтитры: Вкл', 'btn_subs_off': 'Субтитры: Выкл',
        'btn_shadows_on': 'Тени: Вкл', 'btn_shadows_off': 'Тени: Выкл',
        'btn_dyn_density_on': 'Динам. плотность: Вкл', 'btn_dyn_density_off': 'Динам. плотность: Выкл',
        'btn_culling_on': 'Скрытие частиц: Вкл', 'btn_culling_off': 'Скрытие частиц: Выкл',
        'btn_fdm_opt_on': 'Опт. FDM: Вкл', 'btn_fdm_opt_off': 'Опт. FDM: Выкл',
        'btn_full_reset': 'ПОЛНЫЙ СБРОС',
        'label_fdm_progress': 'Прогресс FDM шага:',
        'mode_modal': 'Режим: Модальный', 'mode_point': 'Режим: Точечный (частота)', 'mode_audio': 'Режим: Аудиофайл', 'mode_mic': 'Режим: Микрофон', 'mode_piano': 'Режим: Пианино', 'mode_desktop_audio': 'Режим: Захват Аудио',
        'legend_plate_rotation': 'Вращение пластины',
        'label_rotation_speed': 'Скорость вращения:',
        'btn_stop_rotation': 'Остановить вращение',
        'legend_audio_input': 'Аудиовход',
        'label_audio_file': 'Загрузить аудиофайл(ы):',
        'btn_play_track': 'Воспр. трек/плейлист',
        'btn_stop_audio': 'Остановить аудио',
        'btn_pause': 'Пауза', 'btn_resume': 'Продолжить',
        'btn_prev_track': 'Пред. трек',
        'btn_next_track': 'След. трек',
        'audio_info_none': 'Аудио не загружено',
        'btn_mic_on': 'Микрофон: Вкл', 'btn_mic_off': 'Микрофон: Выкл',
        'mic_info': 'Для использования микрофона может потребоваться разрешение браузера.',
        'label_audio_progress': 'Прогресс аудио:',
        'pitch_detected': 'Определено:', 'pitch_note': 'Нота:', 'pitch_detune': 'Отклонение:',
        'bpm_confidence': 'Увер:',
        'legend_piano': 'Виртуальное Пианино',
        'label_octave': 'Октава:',
        'octave_2': 'Контроктава', 'octave_3': 'Малая', 'octave_4': 'Первая', 'octave_5': 'Вторая', 'octave_6': 'Третья',
        'piano_inactive': 'Пианино неактивно',
        'piano_hint': 'Клавиатура: Ср. ряд - белые, Верх. ряд - черные. Shift + Клавиша = Октава выше.',
        'adv_title': 'Продвинутые настройки',
        'legend_plate_physics': 'Физика пластины',
        'adv_plate_thickness': 'Толщина (м):', 'adv_plate_density': 'Плотность (кг/м³):', 'adv_emodulus': 'Модуль Юнга (Па):', 'adv_poisson': 'Коэф. Пуассона:',
        'legend_fdm_settings': 'Настройки FDM',
        'adv_min_grid': 'Мин. размер сетки:', 'adv_max_grid': 'Макс. размер сетки:', 'adv_fdm_steps': 'Баз. макс. шагов FDM/кадр:', 'adv_stability_factor': 'Фактор стабильности FDM (dt):', 'adv_damping_factor': 'Фактор затухания FDM:',
        'legend_audio_analysis': 'Настройки Аудиоанализа',
        'adv_bpm_window': 'Окно детекции пиков BPM:',
        'legend_particle_dynamics': 'Динамика частиц',
        'adv_particle_count': 'Кол-во частиц (макс.):', 'adv_force_multiplier': 'Множитель силы (база):', 'adv_damping_base': 'Демпфирование (база):', 'adv_particle_size': 'Размер частиц:', 'adv_enable_repulsion': 'Вкл. отталкивание частиц:', 'adv_repulsion_radius': 'Радиус отталкивания:', 'adv_repulsion_strength': 'Сила отталкивания:', 'adv_restitution': 'Упругость столкновения:', 'adv_max_speed': 'Макс. скорость частиц:', 'adv_max_neighbors': 'Макс. соседей для отталк.:',
        'legend_excitation_amp': 'Амплитуда возбуждения',
        'adv_base_amp': 'Базовая амплитуда:', 'adv_low_cutoff': 'Нижний порог частоты (Гц):', 'adv_high_cutoff': 'Верхний порог частоты (Гц):', 'adv_max_factor': 'Макс. множитель:', 'adv_min_factor': 'Мин. множитель:',
        'legend_particle_viz': 'Визуализация частиц',
        'adv_deform_scale': 'Масштаб "прыжка":', 'adv_max_amp': 'Макс. "прыжок":',
        'btn_reset_adv': 'Сбросить продвинутые настройки'
    },
    'en': {
        'welcome_title': 'Chladni Simulator v7.3.55',
        'welcome_body': `
            <p>I took a shit. This is an interactive simulator based on the practical studies of Chladni and other motherfuckers like Bessel roots and other dudes.</p>
            <p>The project is provided under the <b>"I DON'T GIVE A FUCK"</b> license, meaning it's free since it's a hodgepodge of all practical works on the internet. My task was just to bring all this bullshit together, which I basically did, thanks to two AIs — Gemini and Grok. The project could have been more ambitious, but for now, I'm limited to 1,000,000 tokens, though that was enough.</p>
            <p>There are many practical applications, as the project is a unique interactive motherfucker from the perspective of Physics and other Algebraic stuff.</p>
            <p>Special thanks to: Daniil Dmitrievich =)</p>
            <p>4 contact with me: <b>https://t.me/unknown_sector</b></p>
            <p><b>What I haven't done yet, but will do:</b><br>
            - Move to GitHub and deploy on GitHub Pages. (DONE)<br>
            - Port to Electron to enable project implementation using GPU (Win/Linux).<br>
            - Integrate Client-Server functionality (currently, everything runs on the client, which burdens the user's resources (CPU)).</p>
            <p>In the future, when I have time, I'll continue to work on the project, but for now, I'm hiding from the tax authorities. However, donations are always welcome, so here is my TON wallet address on Telegram: <b>UQAY5_p2plWbshvrITsPk5TJ4CKBneJImdVUV-8-MBOA4Lhh</b></p>
            <div class="welcome-image-container">
                <img src="./img/111.png" alt="Chladni Pattern Example" border="0" />
                <img src="./img/222.jpg" alt="Project Logo" border="0" />
            </div>
        `,
        'lang_toggle': 'Переключить на Русский',
        'show_prompt': 'Project Prompt',
        'close_welcome': 'Start',
        'prompt_title': 'Project Prompt / Technical Task',
        'close_modal': 'Close',
        'toggle_left_panel_show': 'Hide left panel',
        'toggle_left_panel_hide': 'Show left panel',
        'toggle_right_panel_show': 'Hide right panel',
        'toggle_right_panel_hide': 'Show right panel',
        'btn_show_welcome': 'Help / Помощь',
        'info_controls': 'Hold Left Mouse Button and move to rotate the camera.<br>Use the Mouse Wheel to zoom in/out.',
        'info_version': 'Simulator v7.3.55 (Interactive Learning).',
        'legend_main_params': 'Main Parameters',
        'label_freq_slider': 'Frequency (slider):',
        'label_freq_input': 'Frequency (Hz input):',
        'btn_set': 'Set',
        'label_sim_speed': 'Simulation Speed:',
        'label_presets': 'Presets (m,n for circle):',
        'preset_custom': 'Custom m, n (now: {m},{n})',
        'preset_custom_inactive': 'Custom m, n (inactive)',
        'preset_lada': 'Star of Lada (4,2)',
        'param_m': 'Mode m:',
        'btn_set_m': 'Set m',
        'param_n': 'Mode n:',
        'btn_set_n': 'Set n',
        'legend_sim_controls': 'Simulation Controls',
        'btn_desktop_audio_on': 'Capture Audio: On', 'btn_desktop_audio_off': 'Capture Audio: Off',
        'btn_sound_on': 'Sound: On', 'btn_sound_off': 'Sound: Off',
        'btn_freeze_on': 'Particles: Frozen', 'btn_freeze_off': 'Particles: Move',
        'btn_subs_on': 'Subtitles: On', 'btn_subs_off': 'Subtitles: Off',
        'btn_shadows_on': 'Shadows: On', 'btn_shadows_off': 'Shadows: Off',
        'btn_dyn_density_on': 'Dyn. Density: On', 'btn_dyn_density_off': 'Dyn. Density: Off',
        'btn_culling_on': 'Culling: On', 'btn_culling_off': 'Culling: Off',
        'btn_fdm_opt_on': 'FDM Opt: On', 'btn_fdm_opt_off': 'FDM Opt: Off',
        'btn_full_reset': 'FULL RESET',
        'label_fdm_progress': 'FDM Step Progress:',
        'mode_modal': 'Mode: Modal', 'mode_point': 'Mode: Point (Frequency)', 'mode_audio': 'Mode: Audio File', 'mode_mic': 'Mode: Microphone', 'mode_piano': 'Mode: Piano', 'mode_desktop_audio': 'Mode: Audio Capture',
        'legend_plate_rotation': 'Plate Rotation',
        'label_rotation_speed': 'Rotation Speed:',
        'btn_stop_rotation': 'Stop Rotation',
        'legend_audio_input': 'Audio Input',
        'label_audio_file': 'Load audio file(s):',
        'btn_play_track': 'Play Track/Playlist',
        'btn_stop_audio': 'Stop Audio',
        'btn_pause': 'Pause', 'btn_resume': 'Resume',
        'btn_prev_track': 'Prev. Track',
        'btn_next_track': 'Next Track',
        'audio_info_none': 'No audio loaded',
        'btn_mic_on': 'Microphone: On', 'btn_mic_off': 'Microphone: Off',
        'mic_info': 'Using the microphone may require browser permission.',
        'label_audio_progress': 'Audio Progress:',
        'pitch_detected': 'Detected:', 'pitch_note': 'Note:', 'pitch_detune': 'Detune:',
        'bpm_confidence': 'Conf:',
        'legend_piano': 'Virtual Piano',
        'label_octave': 'Octave:',
        'octave_2': 'Contra', 'octave_3': 'Small', 'octave_4': 'First', 'octave_5': 'Second', 'octave_6': 'Third',
        'piano_inactive': 'Piano inactive',
        'piano_hint': 'Keyboard: Middle row - white, Top row - black. Shift + Key = Octave up.',
        'adv_title': 'Advanced Settings',
        'legend_plate_physics': 'Plate Physics',
        'adv_plate_thickness': 'Thickness (m):', 'adv_plate_density': 'Density (kg/m³):', 'adv_emodulus': 'Young\'s Modulus (Pa):', 'adv_poisson': 'Poisson\'s Ratio:',
        'legend_fdm_settings': 'FDM Settings',
        'adv_min_grid': 'Min. Grid Size:', 'adv_max_grid': 'Max. Grid Size:', 'adv_fdm_steps': 'Base Max FDM Steps/Frame:', 'adv_stability_factor': 'FDM Stability Factor (dt):', 'adv_damping_factor': 'FDM Damping Factor:',
        'legend_audio_analysis': 'Audio Analysis Settings',
        'adv_bpm_window': 'BPM Peak Detection Window:',
        'legend_particle_dynamics': 'Particle Dynamics',
        'adv_particle_count': 'Particle Count (Max):', 'adv_force_multiplier': 'Force Multiplier (Base):', 'adv_damping_base': 'Damping (Base):', 'adv_particle_size': 'Particle Size:', 'adv_enable_repulsion': 'Enable Repulsion:', 'adv_repulsion_radius': 'Repulsion Radius:', 'adv_repulsion_strength': 'Repulsion Strength:', 'adv_restitution': 'Collision Restitution:', 'adv_max_speed': 'Max Particle Speed:', 'adv_max_neighbors': 'Max Repulsion Neighbors:',
        'legend_excitation_amp': 'Excitation Amplitude',
        'adv_base_amp': 'Base Amplitude:', 'adv_low_cutoff': 'Low Freq. Cutoff (Hz):', 'adv_high_cutoff': 'High Freq. Cutoff (Hz):', 'adv_max_factor': 'Max. Multiplier:', 'adv_min_factor': 'Min. Multiplier:',
        'legend_particle_viz': 'Particle Visualization',
        'adv_deform_scale': '"Jump" Scale:', 'adv_max_amp': 'Max. "Jump":',
        'btn_reset_adv': 'Reset Advanced Settings'
    }
};

const TOOLTIP_TEXTS = {
    'desktop_audio': {
        ru: { title: 'Захват Аудио с ПК', body: 'Позволяет использовать любой звук, воспроизводимый на вашем компьютере (из другой вкладки браузера, плеера, игры) как источник для симуляции. <b>КРИТИЧЕСКИ ВАЖНО:</b> в появившемся окне выберите вкладку "Весь экран" или конкретное окно приложения, а затем <b>обязательно поставьте галочку "Поделиться системным аудио"</b> в левом нижнем углу.' },
        en: { title: 'Desktop Audio Capture', body: 'Allows using any sound playing on your computer (from another browser tab, player, game) as the source for the simulation. <b>CRITICALLY IMPORTANT:</b> In the dialog that appears, select the "Entire Screen" tab or a specific application window, and then <b>be sure to check the "Share system audio" box</b> in the lower-left corner.' }
    },
    'frequency': {
        ru: { title: 'Частота (Ползунок)', body: 'Основной способ находить резонансные частоты. Плавно изменяйте частоту возбуждения, чтобы увидеть, как узоры Хладни появляются и исчезают. Каждому стабильному узору соответствует своя резонансная частота.' },
        en: { title: 'Frequency (Slider)', body: 'The primary way to find resonant frequencies. Smoothly change the excitation frequency to see how Chladni patterns appear and disappear. Each stable pattern corresponds to its own resonant frequency.' }
    },
    'frequency_input': {
        ru: { title: 'Точный Ввод Частоты', body: 'Используйте это поле для ввода точного значения частоты в Герцах (Гц). Это полезно, если вы знаете конкретную резонансную частоту, которую хотите воспроизвести, или для возврата к ранее найденному узору.' },
        en: { title: 'Precise Frequency Input', body: 'Use this field to enter an exact frequency value in Hertz (Hz). This is useful if you know a specific resonant frequency you want to reproduce or to return to a previously found pattern.' }
    },
    'simulation_speed': {
        ru: { title: 'Скорость Симуляции', body: 'Управляет скоростью течения времени в симуляции. Значение > 1.0 ускоряет движение частиц и формирование узора, а значение < 1.0 — замедляет. Не влияет на частоту звука, только на визуальную динамику.' },
        en: { title: 'Simulation Speed', body: 'Controls the flow of time in the simulation. A value > 1.0 speeds up particle movement and pattern formation, while a value < 1.0 slows it down. It does not affect the sound frequency, only the visual dynamics.' }
    },
    'presets': {
        ru: { title: 'Предустановки Узоров', body: 'Быстрый способ выбрать классические узоры Хладни для круглой пластины. Каждая опция устанавливает соответствующие модальные параметры "m" и "n", которые однозначно определяют геометрию узора и его резонансную частоту.' },
        en: { title: 'Pattern Presets', body: 'A quick way to select classic Chladni patterns for a circular plate. Each option sets the corresponding modal parameters "m" and "n", which uniquely define the pattern\'s geometry and its resonant frequency.' }
    },
    'param_m': {
        ru: { title: 'Модальный параметр "m"', body: 'Определяет количество диаметральных узловых линий (линий, проходящих через центр), которые делят узор на сектора. "m" должно быть целым неотрицательным числом (0, 1, 2...). Увеличивайте "m" для получения более "лучистых" узоров.' },
        en: { title: 'Modal Parameter "m"', body: 'Determines the number of diametral nodal lines (lines passing through the center) that divide the pattern into sectors. "m" must be a non-negative integer (0, 1, 2...). Increase "m" to get more "ray-like" patterns.' }
    },
    'param_n': {
        ru: { title: 'Модальный параметр "n"', body: 'Определяет количество круговых узловых линий. "n" должно быть целым положительным числом (1, 2, 3...). Увеличение "n" добавляет концентрические кольца, на которых частицы остаются в покое.' },
        en: { title: 'Modal Parameter "n"', body: 'Determines the number of circular nodal lines. "n" must be a positive integer (1, 2, 3...). Increasing "n" adds concentric rings where particles remain at rest.' }
    },
    'fdm_progress': {
        ru: { title: 'Прогресс Шага Симуляции (FDM)', body: 'Показывает, насколько интенсивно центральный процессор (CPU) рассчитывает колебания пластины в данный момент. Чем выше частота, тем больше вычислений требуется, и тем активнее будет этот индикатор.' },
        en: { title: 'Simulation Step Progress (FDM)', body: 'Shows how intensively the central processor (CPU) is calculating the plate\'s vibrations at the moment. The higher the frequency, the more calculations are required, and the more active this indicator will be.' }
    },
    'rotation_speed': {
        ru: { title: 'Скорость Вращения Пластины', body: 'Заставляет пластину вращаться вокруг своей оси. Это чисто визуальный эффект, который не влияет на физику узоров, но позволяет лучше рассмотреть их со всех сторон.' },
        en: { title: 'Plate Rotation Speed', body: 'Causes the plate to rotate around its axis. This is a purely visual effect that does not affect the physics of the patterns but allows you to view them from all sides.' }
    },
    'audio_file': {
        ru: { title: 'Анализ Аудиофайла', body: 'Загрузите аудиофайл (MP3, WAV и др.) для анализа. Симулятор будет в реальном времени определять основную частоту звука и использовать ее для возбуждения пластины. Также будет предпринята попытка найти текст песни (субтитры).' },
        en: { title: 'Audio File Analysis', body: 'Load an audio file (MP3, WAV, etc.) for analysis. The simulator will determine the fundamental frequency of the sound in real-time and use it to excite the plate. It will also attempt to find the song lyrics (subtitles).' }
    },
    'mic_input': {
        ru: { title: 'Анализ с Микрофона', body: 'Используйте звук с вашего микрофона для управления симуляцией. Попробуйте напеть ноту или поднести источник звука к микрофону, чтобы увидеть, как узоры реагируют на ваш голос или музыку.' },
        en: { title: 'Microphone Analysis', body: 'Use the sound from your microphone to control the simulation. Try singing a note or bringing a sound source close to the microphone to see how the patterns react to your voice or music.' }
    },
    'audio_progress': {
        ru: { title: 'Прогресс Воспроизведения', body: 'Показывает текущую позицию в загруженном аудиофайле. Вы можете перематывать трек, перетаскивая этот ползунок.' },
        en: { title: 'Playback Progress', body: 'Shows the current position in the loaded audio file. You can seek through the track by dragging this slider.' }
    },
    'piano_octave': {
        ru: { title: 'Выбор Октавы', body: 'Изменяет высоту нот, играемых на виртуальном пианино и с клавиатуры компьютера. "Первая октава" является стандартной для фортепиано.' },
        en: { title: 'Octave Selection', body: 'Changes the pitch of the notes played on the virtual piano and with the computer keyboard. The "First Octave" is standard for piano.' }
    },
    'piano_keys': {
        ru: { title: 'Виртуальное Пианино', body: 'Нажимайте на клавиши мышью или используйте клавиатуру компьютера (средний ряд - белые, верхний - черные), чтобы генерировать узоры, соответствующие определенным нотам. Удерживайте Shift для игры на октаву выше.' },
        en: { title: 'Virtual Piano', body: 'Click the keys with your mouse or use the computer keyboard (middle row - white keys, top row - black keys) to generate patterns corresponding to specific notes. Hold Shift to play one octave higher.' }
    },
    'adv_plate_thickness': {
        ru: { title: 'Толщина Пластины', body: 'Физическая толщина пластины в метрах. Более толстые пластины жестче, что приводит к увеличению скорости распространения волн и, как следствие, к более высоким резонансным частотам для тех же узоров (m,n).' },
        en: { title: 'Plate Thickness', body: 'The physical thickness of the plate in meters. Thicker plates are stiffer, which leads to a higher wave propagation speed and, consequently, higher resonant frequencies for the same patterns (m,n).' }
    },
    'adv_plate_density': {
        ru: { title: 'Плотность Материала', body: 'Плотность материала пластины в кг/м³. Более плотные материалы (например, сталь по сравнению с алюминием) более инертны. Это замедляет волны и понижает резонансные частоты.' },
        en: { title: 'Material Density', body: 'The density of the plate material in kg/m³. Denser materials (e.g., steel compared to aluminum) are more inert. This slows down the waves and lowers the resonant frequencies.' }
    },
    'adv_emodulus': {
        ru: { title: 'Модуль Юнга', body: 'Характеристика упругости материала, показывающая его жесткость. Чем выше модуль Юнга (например, у стали он выше, чем у алюминия), тем быстрее распространяются волны и тем выше резонансные частоты.' },
        en: { title: 'Young\'s Modulus', body: 'A measure of a material\'s stiffness. The higher the Young\'s modulus (e.g., steel is higher than aluminum), the faster the waves propagate and the higher the resonant frequencies.' }
    },
    'adv_poisson': {
        ru: { title: 'Коэффициент Пуассона', body: 'Описывает, насколько материал сжимается в поперечном направлении при растяжении в продольном. Влияет на изгибную жесткость пластины и незначительно изменяет резонансные частоты.' },
        en: { title: 'Poisson\'s Ratio', body: 'Describes how much a material compresses in the transverse direction when stretched longitudinally. It affects the flexural rigidity of the plate and slightly alters the resonant frequencies.' }
    },
    'adv_min_grid': {
        ru: { title: 'Мин. Размер Сетки', body: 'Минимальный размер расчетной сетки (например, 31x31), используемой для низких частот. Меньшие значения снижают нагрузку на CPU, но могут ухудшить детализацию узоров.' },
        en: { title: 'Min. Grid Size', body: 'The minimum size of the calculation grid (e.g., 31x31) used for low frequencies. Smaller values reduce CPU load but may decrease pattern detail.' }
    },
    'adv_max_grid': {
        ru: { title: 'Макс. Размер Сетки', body: 'Максимальный размер расчетной сетки (например, 121x121), используемой для высоких частот. Более высокие значения позволяют точно моделировать короткие волны, но значительно увеличивают нагрузку на CPU.' },
        en: { title: 'Max. Grid Size', body: 'The maximum size of the calculation grid (e.g., 121x121) used for high frequencies. Larger values allow for accurate modeling of short waves but significantly increase CPU load.' }
    },
    'adv_fdm_steps': {
        ru: { title: 'Шаги FDM / Кадр', body: 'Базовое количество шагов численного моделирования, выполняемых за один кадр анимации. Большее число шагов повышает стабильность симуляции на высоких частотах, но требует больше ресурсов.' },
        en: { title: 'FDM Steps / Frame', body: 'The base number of numerical simulation steps performed per animation frame. A higher number of steps improves simulation stability at high frequencies but requires more resources.' }
    },
    'adv_stability_factor': {
        ru: { title: 'Фактор Стабильности (dt)', body: 'Множитель для расчета шага по времени (dt) в симуляции. Меньшие значения делают симуляцию более стабильной, но медленной, так как требуется больше шагов. Слишком большие значения могут привести к "взрыву" симуляции.' },
        en: { title: 'Stability Factor (dt)', body: 'A multiplier for calculating the time step (dt) in the simulation. Smaller values make the simulation more stable but slower, as more steps are required. Values that are too large can cause the simulation to "explode".' }
    },
    'adv_damping_factor': {
        ru: { title: 'Затухание FDM', body: 'Имитирует внутреннее трение в материале пластины, заставляя колебания со временем затухать. Слишком низкое значение может привести к нестабильности, а слишком высокое - "заглушит" узоры.' },
        en: { title: 'FDM Damping', body: 'Simulates internal friction in the plate material, causing vibrations to decay over time. A value that is too low can lead to instability, while a value that is too high will "dampen" the patterns.' }
    },
    'adv_bpm_window': {
        ru: { title: 'Окно Детекции BPM', body: 'Размер "окна" истории (в сэмплах), используемого для поиска пиков при анализе ритма (BPM). Увеличение может сделать определение более стабильным, но менее отзывчивым к смене темпа.' },
        en: { title: 'BPM Detection Window', body: 'The size of the history "window" (in samples) used to find peaks when analyzing the rhythm (BPM). Increasing it can make detection more stable but less responsive to tempo changes.' }
    },
    'adv_particle_count': {
        ru: { title: 'Количество Частиц', body: 'Максимальное количество частиц (песка) в симуляции. Большее количество создает более четкие и детализированные узоры, но сильно влияет на производительность.' },
        en: { title: 'Particle Count', body: 'The maximum number of particles (sand) in the simulation. A larger quantity creates clearer and more detailed patterns but heavily impacts performance.' }
    },
    'adv_force_multiplier': {
        ru: { title: 'Множитель Силы', body: 'Базовый множитель для силы, которая отталкивает частицы от вибрирующих участков к узловым линиям. Увеличьте, чтобы частицы двигались быстрее и энергичнее.' },
        en: { title: 'Force Multiplier', body: 'The base multiplier for the force that pushes particles from vibrating areas to the nodal lines. Increase it to make the particles move faster and more energetically.' }
    },
    'adv_damping_base': {
        ru: { title: 'Демпфирование Частиц', body: 'Имитирует трение, замедляющее частицы (значение < 1.0). Чем ближе к 1, тем дольше частицы сохраняют свою скорость. Слишком низкие значения заставят их быстро остановиться.' },
        en: { title: 'Particle Damping', body: 'Simulates friction that slows down the particles (value < 1.0). The closer to 1, the longer the particles retain their velocity. Values that are too low will cause them to stop quickly.' }
    },
    'adv_particle_size': {
        ru: { title: 'Размер Частиц', body: 'Визуальный размер каждой частицы. Не влияет на физику, только на отображение. Подберите размер для наилучшего визуального результата.' },
        en: { title: 'Particle Size', body: 'The visual size of each particle. Does not affect the physics, only the display. Adjust the size for the best visual result.' }
    },
    'adv_enable_repulsion': {
        ru: { title: 'Отталкивание Частиц', body: 'Включает/выключает силу отталкивания между частицами. Когда включено, частицы ведут себя более "жидко" и не скапливаются в одной точке, распределяясь вдоль узловых линий.' },
        en: { title: 'Particle Repulsion', body: 'Enables/disables the repulsive force between particles. When enabled, particles behave more "fluidly" and do not clump together at a single point, distributing themselves along the nodal lines.' }
    },
    'adv_repulsion_radius': {
        ru: { title: 'Радиус Отталкивания', body: 'Расстояние, на котором частицы начинают отталкивать друг друга. Больший радиус создает более "разреженные" узоры.' },
        en: { title: 'Repulsion Radius', body: 'The distance at which particles begin to repel each other. A larger radius creates more "sparse" patterns.' }
    },
    'adv_repulsion_strength': {
        ru: { title: 'Сила Отталкивания', body: 'Насколько сильно частицы отталкиваются друг от друга в пределах радиуса. Увеличьте для более выраженного эффекта "жидкости".' },
        en: { title: 'Repulsion Strength', body: 'How strongly particles repel each other within the radius. Increase for a more pronounced "fluid" effect.' }
    },
    'adv_restitution': {
        ru: { title: 'Упругость Столкновения', body: 'Коэффициент упругости (0 до 1) при столкновении частиц с краем пластины. 0 - абсолютно неупругое столкновение (частица "прилипает"), 1 - абсолютно упругое (отскакивает без потерь).' },
        en: { title: 'Collision Restitution', body: 'The coefficient of restitution (0 to 1) for particle collisions with the edge of the plate. 0 is a perfectly inelastic collision (particle "sticks"), 1 is a perfectly elastic collision (bounces with no loss).' }
    },
    'adv_max_speed': {
        ru: { title: 'Макс. Скорость Частиц', body: 'Ограничивает максимальную скорость движения частиц, предотвращая их "вылет" из-за слишком больших сил.' },
        en: { title: 'Max Particle Speed', body: 'Limits the maximum speed of particle movement, preventing them from "flying out" due to excessively large forces.' }
    },
    'adv_max_neighbors': {
        ru: { title: 'Макс. Соседей для Отталкивания', body: 'Ограничивает количество соседних частиц, проверяемых для расчета отталкивания. Уменьшение этого значения может повысить производительность, но снизит качество симуляции отталкивания.' },
        en: { title: 'Max Repulsion Neighbors', body: 'Limits the number of neighboring particles checked for calculating repulsion. Decreasing this value can improve performance but will reduce the quality of the repulsion simulation.' }
    },
    'adv_base_amp': {
        ru: { title: 'Базовая Амплитуда', body: 'Основной множитель для силы возбуждения пластины. Увеличение этого параметра делает колебания более сильными, что приводит к более быстрому формированию узоров.' },
        en: { title: 'Base Amplitude', body: 'The main multiplier for the plate excitation force. Increasing this parameter makes the vibrations stronger, leading to faster pattern formation.' }
    },
    'adv_low_cutoff': {
        ru: { title: 'Нижний Порог Частоты', body: 'Частота, ниже которой амплитуда возбуждения будет максимальной. Используется для усиления низких частот.' },
        en: { title: 'Low Frequency Cutoff', body: 'The frequency below which the excitation amplitude will be at its maximum. Used to boost low frequencies.' }
    },
    'adv_high_cutoff': {
        ru: { title: 'Верхний Порог Частоты', body: 'Частота, выше которой амплитуда возбуждения будет минимальной. Используется для ослабления очень высоких частот и стабилизации симуляции.' },
        en: { title: 'High Frequency Cutoff', body: 'The frequency above which the excitation amplitude will be at its minimum. Used to attenuate very high frequencies and stabilize the simulation.' }
    },
    'adv_max_factor': {
        ru: { title: 'Макс. Множитель Амплитуды', body: 'Множитель, применяемый к базовой амплитуде на частотах ниже нижнего порога. Позволяет усилить отклик на низких частотах.' },
        en: { title: 'Max. Amplitude Factor', body: 'The multiplier applied to the base amplitude for frequencies below the low cutoff. Allows for boosting the response at low frequencies.' }
    },
    'adv_min_factor': {
        ru: { title: 'Мин. Множитель Амплитуды', body: 'Множитель, применяемый к базовой амплитуде на частотах выше верхнего порога. Позволяет ослабить отклик на высоких частотах.' },
        en: { title: 'Min. Amplitude Factor', body: 'The multiplier applied to the base amplitude for frequencies above the high cutoff. Allows for attenuating the response at high frequencies.' }
    },
    'adv_deform_scale': {
        ru: { title: 'Масштаб "Прыжка"', body: 'Визуальный множитель, который управляет тем, насколько высоко "подпрыгивают" частицы над вибрирующими участками. Не влияет на физику, только на визуализацию.' },
        en: { title: 'Jump" Scale', body: 'A visual multiplier that controls how high the particles "jump" over the vibrating areas. Does not affect the physics, only the visualization.' }
    },
    'adv_max_amp': {
        ru: { title: 'Макс. "Прыжок"', body: 'Ограничивает максимальную высоту "прыжка" частиц. Предотвращает чрезмерно большие визуальные смещения, которые могут выглядеть неестественно.' },
        en: { title: 'Max. "Jump"', body: 'Limits the maximum height of the particle "jump". Prevents excessively large visual displacements that can look unnatural.' }
    }
};

const PROJECT_PROMPT_TEXT = '';

const DEBUG_MODE = false;

const PLATE_RADIUS_DEFAULT = 7.5;
const PLATE_THICKNESS_DEFAULT = 0.002;
const PLATE_DENSITY_DEFAULT = 7850;
const E_MODULUS_DEFAULT = 200e9;
const POISSON_RATIO_DEFAULT = 0.3;
const PARTICLE_COUNT_DEFAULT = 9000;
const PARTICLE_FORCE_BASE_DEFAULT = 1.5e6;
const PARTICLE_DAMPING_BASE_DEFAULT = 0.95; 
const PARTICLE_RESTITUTION_DEFAULT = 0.5;
const MAX_PARTICLE_SPEED_DEFAULT = 18.0;
const PARTICLE_SIZE_DEFAULT = 0.049; 
const ENABLE_REPULSION_DEFAULT = true;
const REPULSION_RADIUS_DEFAULT = 0.15;
const REPULSION_STRENGTH_DEFAULT = 0.005;
const MAX_REPULSION_NEIGHBORS_DEFAULT = 50;
const MIN_GRID_SIZE_DEFAULT = 33;
const MAX_GRID_SIZE_DEFAULT = 151;
const FDM_STABILITY_FACTOR_DEFAULT = 0.08; 
const FDM_DAMPING_FACTOR_DEFAULT = 0.000050;
const BASE_FDM_STEPS_CPU_DEFAULT = 69;
const MAX_VISUAL_AMPLITUDE_DEFAULT = 0.3;
const VISUAL_DEFORMATION_SCALE_DEFAULT = 50.0;
const EXC_BASE_AMP_DEFAULT = 2.0e4;
const EXC_LOW_CUTOFF_DEFAULT = 100;
const EXC_HIGH_CUTOFF_DEFAULT = 3000;
const EXC_MAX_FACTOR_DEFAULT = 3.0;
const EXC_MIN_FACTOR_DEFAULT = 0.5;

const ENABLE_FDM_OPTIMIZATION_DEFAULT = false;
const ENABLE_SHADOWS_DEFAULT = false;
const ENABLE_DYNAMIC_PARTICLE_DENSITY_DEFAULT = false;
const MIN_DYNAMIC_PARTICLE_COUNT = 2000;
const PARTICLE_COUNT_UPDATE_THROTTLE_MS = 750;
const ENABLE_STUCK_PARTICLE_CULLING_DEFAULT = false;
const STATIONARY_VELOCITY_THRESHOLD = 0.01;
const STUCK_PARTICLE_DISPLACEMENT_THRESHOLD_FACTOR = 0.05;
const ENABLE_SUBTITLES_DEFAULT = true;

const PITCH_MIN_FREQUENCY_HZ = 30;
const PITCH_MAX_FREQUENCY_HZ = 20000; 
const PITCH_UPDATE_INTERVAL_SECONDS = 0.03; 
const PITCH_CHANGE_THRESHOLD_HZ = 8.9;
const PITCH_SMOOTHING_FACTOR = 0.15;

const BPM_PEAK_DETECTION_WINDOW_DEFAULT = 10;
const MAX_AUDIO_DURATION_FOR_MULTIBAND_S = 1800;
const BPM_LOW_BAND_WEIGHT = 3.0;
const BPM_MID_BAND_WEIGHT = 1.5;
const BPM_HIGH_BAND_WEIGHT = 1.2;

const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const NOTE_TO_MIDI_NUMBER_OFFSET = { 'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11 };

class ChladniSimulator {
    constructor(besselRootsTable) {
        this.besselRootsTable = besselRootsTable;
        this.PLATE_RADIUS = PLATE_RADIUS_DEFAULT;
        this.PLATE_THICKNESS = PLATE_THICKNESS_DEFAULT;
        this.PLATE_DENSITY = PLATE_DENSITY_DEFAULT;
        this.E_MODULUS = E_MODULUS_DEFAULT;
        this.POISSON_RATIO = POISSON_RATIO_DEFAULT;
        this.D_FLEXURAL_RIGIDITY = 0;
        this.RHO_H_PLATE_SPECIFIC_DENSITY = 0;
        this.PARTICLE_COUNT = PARTICLE_COUNT_DEFAULT;
        this.MAX_PARTICLE_COUNT_USER_SETTING = PARTICLE_COUNT_DEFAULT;
        this.PARTICLE_EFFECTIVE_FORCE_MULTIPLIER_BASE = PARTICLE_FORCE_BASE_DEFAULT;
        this.PARTICLE_DAMPING_BASE = PARTICLE_DAMPING_BASE_DEFAULT;
        this.PARTICLE_BOUNDARY_RESTITUTION = PARTICLE_RESTITUTION_DEFAULT;
        this.MAX_PARTICLE_SPEED = MAX_PARTICLE_SPEED_DEFAULT;
        this.PARTICLE_SIZE = PARTICLE_SIZE_DEFAULT;
        this.ENABLE_PARTICLE_REPULSION = ENABLE_REPULSION_DEFAULT;
        this.PARTICLE_REPULSION_RADIUS = REPULSION_RADIUS_DEFAULT;
        this.PARTICLE_REPULSION_STRENGTH = REPULSION_STRENGTH_DEFAULT;
        this.MAX_REPULSION_NEIGHBORS_CHECK = MAX_REPULSION_NEIGHBORS_DEFAULT;
        this.currentGridSize = MIN_GRID_SIZE_DEFAULT;
        this.MIN_GRID_SIZE = MIN_GRID_SIZE_DEFAULT;
        this.MAX_GRID_SIZE = MAX_GRID_SIZE_DEFAULT;
        this.FDM_STABILITY_FACTOR = FDM_STABILITY_FACTOR_DEFAULT;
        this.FDM_DAMPING_FACTOR = FDM_DAMPING_FACTOR_DEFAULT;
        this.BASE_MAX_FDM_STEPS_PER_FRAME = BASE_FDM_STEPS_CPU_DEFAULT;
        this.enableFDMOptimization = ENABLE_FDM_OPTIMIZATION_DEFAULT;
        this.enableShadows = ENABLE_SHADOWS_DEFAULT;
        this.enableDynamicParticleDensity = ENABLE_DYNAMIC_PARTICLE_DENSITY_DEFAULT;
        this.enableStuckParticleCulling = ENABLE_STUCK_PARTICLE_CULLING_DEFAULT;
        this.isSubtitlesEnabled = ENABLE_SUBTITLES_DEFAULT;
        this.lastParticleCountUpdateTime = 0;
        this.fdmConfiguredFrequency = 0;
        this.MAX_VISUAL_AMPLITUDE = MAX_VISUAL_AMPLITUDE_DEFAULT;
        this.VISUAL_DEFORMATION_SCALE = VISUAL_DEFORMATION_SCALE_DEFAULT;
        this.EXCITATION_FREQ_DEP_BASE_AMP = EXC_BASE_AMP_DEFAULT;
        this.EXCITATION_FREQ_DEP_LOW_CUTOFF = EXC_LOW_CUTOFF_DEFAULT;
        this.EXCITATION_FREQ_DEP_HIGH_CUTOFF = EXC_HIGH_CUTOFF_DEFAULT;
        this.EXCITATION_FREQ_DEP_MAX_FACTOR = EXC_MAX_FACTOR_DEFAULT;
        this.EXCITATION_FREQ_DEP_MIN_FACTOR = EXC_MIN_FACTOR_DEFAULT;
        this.normalExcBaseAmp = EXC_BASE_AMP_DEFAULT;
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
        this.isLoadingTrack = false;
        this.activeFetchID = 0;
        this.scene = null; this.camera = null; this.renderer = null;
        this.orbitControls = null; this.particlesMesh = null;
        this.particleData = []; this.particleInstanceMatrix = new THREE.Matrix4();
        this.gpuComputeManager = null;
        this.useGPUComputing = true; // Флаг для переключения между CPU и GPU
        this.animationClock = new THREE.Clock();
        this.groundPlane = null; this.dirLight1 = null; this.dirLight2 = null;
        this.u_curr_cpu_array2D = null; this.u_prev_cpu_array2D = null; this.u_next_temp_cpu_array2D = null;
        this.modalExcitationPattern_cpu_array2D = null;
        this.plateWidth = 0; this.plateHeight = 0;
        this.dx = 0; this.dy = 0; this.dt_simulation_step = 0;
        this.mainAudioContext = null; this.generatedSoundOscillator = null; this.generatedSoundGainNode = null;
        this.isGeneratedSoundEnabled = false;
        this.playlistFiles = [];
        this.currentPlaylistIndex = -1;
        this.audioElement = null;
        this.audioFileSourceNode = null;
        this.audioFileBuffer = null;
        this.isAudioFilePlaying = false;
        this.isAudioFilePaused = false;
        this.audioFileDuration = 0;
        this.audioPlaybackOffset = 0;
        this.audioFileCurrentTime = 0;
        this.subtitleContainer = null;
        this.currentSubtitles = [];
        this.pitchDetectorAnalyserNode = null;
        this.pitchDetectorSignalBuffer = null;
        this.MIN_SAMPLES_FOR_PITCH_DETECTION = 0;
        this.MAX_SAMPLES_FOR_PITCH_DETECTION = 0;
        this.lastPitchUpdateTime = 0;
        this.lastStablePitchFrequency = 0;
        this.smoothedPitchFrequency = 273;
        this.isMicrophoneEnabled = false;
        this.microphoneStream = null;
        this.microphoneSourceNode = null;
        this.isDesktopAudioEnabled = false;
        this.desktopStream = null;
        this.desktopAudioSourceNode = null;
        this.bpmAnalyzers = {
            low: { analyser: null, history: [], previousData: null, threshold: 100 },
            mid: { analyser: null, history: [], previousData: null, threshold: 100 },
            high: { analyser: null, history: [], previousData: null, threshold: 100 }
        };
        this.bandSources = null;
        this.allDetectedBeats = [];
        this.currentBPM = 120;
        this.bpmUpdateIntervalId = null;
        this.BPM_PEAK_DETECTION_WINDOW = BPM_PEAK_DETECTION_WINDOW_DEFAULT;
        this.fftAnalyserNode = null; this.frequencyData = null;
        this.currentPianoOctave = 4; this.activePianoKeys = new Set(); this.keyboardPressedKeys = new Set();
        this.keyToNoteMapping = {
            'KeyA': 'C', 'KeyS': 'D', 'KeyD': 'E', 'KeyF': 'F',
            'KeyG': 'G', 'KeyH': 'A', 'KeyJ': 'B', 'KeyK': 'C5',
            'KeyW': 'C#', 'KeyE': 'D#', 'KeyT': 'F#', 'KeyY': 'G#', 'KeyU': 'A#'
        };
        this.uiElements = {}; this.defaultAdvancedSettings = {}; this.besselZerosCache = {};
        this.currentLanguage = 'ru';
        this.tooltipTimeout = null;
        this._mainInitialization();
    }
    
    _roundToOddInteger(number) {
        number = Math.max(1, Math.round(number));
        return (number % 2 === 0) ? number + 1 : number;
    }

    _parseInputNumber(value, defaultValue = 0, isInt = false, minVal = -Infinity, maxVal = Infinity) {
        let num = isInt ? parseInt(value) : parseFloat(value);
        if (isNaN(num) || !isFinite(num)) { return defaultValue; }
        return Math.max(minVal, Math.min(maxVal, num));
    }

    _linearInterpolate(startValue, endValue, t) {
        return startValue + (endValue - startValue) * t;
    }

    _formatTimeMMSS(totalSeconds) {
        if (isNaN(totalSeconds) || totalSeconds < 0) return "0:00";
        const minutes = Math.floor(totalSeconds / 60);
        const seconds = Math.floor(totalSeconds % 60).toString().padStart(2, '0');
        return `${minutes}:${seconds}`;
    }

    _factorial(n) {
        if (n < 0) return Infinity;
        if (n === 0) return 1;
        if (n > 170) return Infinity;
        let result = 1;
        for (let i = n; i > 0; i--) result *= i;
        return result;
    }

    _besselJ_fallback(order, xValue) {
        const m = Math.abs(order);
        if (xValue < 0) return Number.isInteger(m) ? ((m % 2 === 0) ? this._besselJ_fallback(m, -xValue) : -this._besselJ_fallback(m, -xValue)) : NaN;
        if (xValue === 0) return m === 0 ? 1.0 : 0.0;
        if (xValue > 30 && xValue > m * 1.5) return Math.sqrt(2 / (Math.PI * xValue)) * Math.cos(xValue - (m * Math.PI / 2) - (Math.PI / 4));
        let sum = 0;
        const termsLimit = Math.max(25, Math.floor(xValue + 15 + m * 0.75));
        for (let k = 0; k <= termsLimit; k++) {
            if (m + k > 170) break;
            const num = Math.pow(-1, k) * Math.pow(xValue / 2, m + 2 * k);
            const den_k_fact = this._factorial(k);
            const den_m_plus_k_fact = this._factorial(m + k);
            if (den_k_fact === Infinity || den_m_plus_k_fact === Infinity || den_k_fact === 0 || den_m_plus_k_fact === 0) continue;
            const term = num / (den_k_fact * den_m_plus_k_fact);
            if (!isFinite(term)) continue;
            sum += term;
        }
        return sum;
    }

    _besselJ_lib(order, xValue) {
        try {
            const pO = parseFloat(order);
            const pX = parseFloat(xValue);
            if (isNaN(pO) || isNaN(pX) || typeof BESSEL === 'undefined' || typeof BESSEL.besselj !== 'function') {
                return this._besselJ_fallback(order, xValue);
            }
            return BESSEL.besselj(pX, pO);
        } catch (e) {
            return this._besselJ_fallback(order, xValue);
        }
    }

    _findNthBesselRootFromTable(mOrder, nthRootOneIndexed) {
        if (mOrder < 0 || nthRootOneIndexed <= 0) return null;
        const m = Math.round(mOrder);
        const n = Math.round(nthRootOneIndexed);
        const n0idx = n - 1;
        if (!this.besselRootsTable) {
            return null;
        }
        if (this.besselRootsTable.hasOwnProperty(m) && n0idx < this.besselRootsTable[m].length) {
            return this.besselRootsTable[m][n0idx];
        }
        return null;
    }

    _findNthBesselRootNumerically(mOrder, nthRootOneIndexed) {
        const precision = 1e-7;
        const maxIter = 100;
        let step = 0.05;
        const maxX = Math.max(250, mOrder + nthRootOneIndexed * Math.PI + 30);
        let rootsFound = 0;
        let currentX = 0.01;
        if (mOrder === 0 && nthRootOneIndexed === 1) currentX = 2.0;
        else if (mOrder > 0 && nthRootOneIndexed === 1) currentX = Math.max(0.01, (mOrder + 1.85575 * Math.pow(mOrder, 1 / 3) - 1.5) * 0.7);
        else if (nthRootOneIndexed > 1) {
            let prevRootApprox = (((nthRootOneIndexed - 2) + mOrder / 2 + 0.75 - 1 / (8 * Math.PI * ((nthRootOneIndexed - 2) + mOrder / 2 + 0.75))) * Math.PI);
            currentX = Math.max(prevRootApprox + step * 0.5, ((nthRootOneIndexed - 1) + mOrder / 2 + 0.25) * Math.PI * 0.7);
        }
        currentX = Math.max(0.01, currentX);
        let prevVal = this._besselJ_lib(mOrder, currentX);
        if (isNaN(prevVal)) {
            currentX += step * 10;
            prevVal = this._besselJ_lib(mOrder, currentX);
            if (isNaN(prevVal)) return null;
        }
        let iterSinceRoot = 0;
        const maxIterNoRoot = 20000;
        while (currentX < maxX && iterSinceRoot < maxIterNoRoot) {
            iterSinceRoot++;
            const nextX = currentX + step;
            const currVal = this._besselJ_lib(mOrder, nextX);
            if (isNaN(currVal)) {
                currentX = nextX + step * 3;
                prevVal = this._besselJ_lib(mOrder, currentX);
                if (isNaN(prevVal)) return null;
                step = Math.min(0.2, step * 1.1);
                continue;
            }
            if ((prevVal * currVal < 0) || (Math.abs(currVal) < precision && Math.abs(prevVal) > precision * 10)) {
                if (!(Math.abs(currVal) < precision && Math.abs(prevVal) < precision && Math.sign(currVal) === Math.sign(prevVal) && rootsFound < nthRootOneIndexed - 1)) {
                    rootsFound++;
                    iterSinceRoot = 0;
                    step = Math.max(1e-7, step * 0.3);
                    if (rootsFound === nthRootOneIndexed) {
                        let low = currentX, high = nextX;
                        if (Math.abs(currVal) < precision) return nextX;
                        for (let i = 0; i < maxIter; i++) {
                            const mid = (low + high) / 2;
                            if (Math.abs(high - low) < precision * 0.1 || mid === low || mid === high) break;
                            const midVal = this._besselJ_lib(mOrder, mid);
                            if (isNaN(midVal)) return Math.abs(this._besselJ_lib(mOrder, low)) < Math.abs(this._besselJ_lib(mOrder, high)) ? low : high;
                            if (Math.abs(midVal) < precision) return mid;
                            let lowVal = this._besselJ_lib(mOrder, low);
                            if (isNaN(lowVal)) lowVal = (midVal > 0 ? -1 : 1) * 1e-9;
                            if (lowVal * midVal <= 0) high = mid; else low = mid;
                        }
                        return (low + high) / 2;
                    }
                }
            }
            prevVal = currVal;
            currentX = nextX;
            if (iterSinceRoot > maxIterNoRoot * 0.7 && step < 0.1) step = Math.min(0.1, step * 1.02);
            else if (step > 0.1) step *= 0.98;
        }
        return null;
    }

    _getBesselZero(mOrder, nthRootOneIndexed) {
        const m = Math.round(mOrder);
        const n = Math.round(nthRootOneIndexed);
        if (m < 0 || n <= 0) return null;
        const cacheKey = `${m},${n}`;
        if (this.besselZerosCache[cacheKey]) return this.besselZerosCache[cacheKey];
        let root = this._findNthBesselRootFromTable(m, n) || this._findNthBesselRootNumerically(m, n);
        if (root !== null && root > 0) {
            this.besselZerosCache[cacheKey] = root;
        } else {
            root = (m === 0 && n === 1) ? 2.4048 : (((n - 1) + m / 2 + 0.75) * Math.PI);
            this.besselZerosCache[cacheKey] = root;
        }
        return root;
    }

    _updatePhysicalConstants() {
        this.D_FLEXURAL_RIGIDITY = (this.E_MODULUS * Math.pow(this.PLATE_THICKNESS, 3)) / (12 * (1 - Math.pow(this.POISSON_RATIO, 2)));
        this.RHO_H_PLATE_SPECIFIC_DENSITY = this.PLATE_DENSITY * this.PLATE_THICKNESS;
        if (!isFinite(this.D_FLEXURAL_RIGIDITY) || this.D_FLEXURAL_RIGIDITY <= 0) {
            this.D_FLEXURAL_RIGIDITY = 100;
        }
        if (!isFinite(this.RHO_H_PLATE_SPECIFIC_DENSITY) || this.RHO_H_PLATE_SPECIFIC_DENSITY <= 0) {
            this.RHO_H_PLATE_SPECIFIC_DENSITY = 10;
        }
    }

    _getOptimalGridSizeForFrequency(frequency) {
        let targetGridSize;
        const baseFreqMin = 200;
        const freqMaxGrid = 7500;
        const effFreq = Math.max(1, frequency);
        if (effFreq <= baseFreqMin) targetGridSize = this.MIN_GRID_SIZE;
        else if (effFreq >= freqMaxGrid) targetGridSize = this.MAX_GRID_SIZE;
        else targetGridSize = this.MIN_GRID_SIZE + ((effFreq - baseFreqMin) / (freqMaxGrid - baseFreqMin)) * (this.MAX_GRID_SIZE - this.MIN_GRID_SIZE);
        return this._roundToOddInteger(Math.max(this.MIN_GRID_SIZE, Math.min(this.MAX_GRID_SIZE, Math.floor(targetGridSize))));
    }

    _getResonantFrequencyKirchhoff(mVal, nVal) {
        const lambda_mn = this._getBesselZero(mVal, nVal);
        if (lambda_mn === null || lambda_mn <= 0) return 20;
        this._updatePhysicalConstants();
        const freq = (Math.pow(lambda_mn / this.PLATE_RADIUS, 2) / (2 * Math.PI)) * Math.sqrt(this.D_FLEXURAL_RIGIDITY / this.RHO_H_PLATE_SPECIFIC_DENSITY);
        return Math.max(1, (isFinite(freq) ? freq : 20));
    }
    
    _setupPlateParametersForCurrentMode() {
        if (this.drivingMechanism === 'modal') {
            this.currentFrequency = this._getResonantFrequencyKirchhoff(this.mParameter, this.nParameter);
            this.actualAppliedFrequency = this.currentFrequency;
        } else if (this.drivingMechanism === 'piano') {
             if (this.activePianoKeys.size > 0 || this.keyboardPressedKeys.size > 0) {
                let sumFreq = 0; let count = 0;
                const allPressedNotes = new Set([...this.activePianoKeys, ...Array.from(this.keyboardPressedKeys).map(kc => this.keyToNoteMapping[kc]).filter(Boolean)]);
                allPressedNotes.forEach(noteNameOrCode => {
                    let note = noteNameOrCode;
                    if (note.startsWith("Key")) note = this.keyToNoteMapping[noteNameOrCode];
                    if (NOTE_TO_MIDI_NUMBER_OFFSET.hasOwnProperty(note) || (note.endsWith('5') && NOTE_TO_MIDI_NUMBER_OFFSET.hasOwnProperty(note.slice(0, -1)))) {
                        let octaveOffset = this.currentPianoOctave;
                        let baseNoteName = note;
                        if (note === 'C5') { octaveOffset = this.currentPianoOctave + 1; baseNoteName = 'C'; }
                        else if (note.length === 2 && note[1] === '5' && NOTE_TO_MIDI_NUMBER_OFFSET.hasOwnProperty(note[0])) {
                            octaveOffset = this.currentPianoOctave + 1; baseNoteName = note[0];
                        }
                        const baseMidiNum = NOTE_TO_MIDI_NUMBER_OFFSET[baseNoteName] + (octaveOffset * 12);
                        sumFreq += this._frequencyFromMIDINoteNumber(baseMidiNum);
                        count++;
                    }
                });
                this.actualAppliedFrequency = count > 0 ? sumFreq / count : this.currentFrequency;
            } else {
                this.actualAppliedFrequency = this.currentFrequency;
            }
        } else if (this.drivingMechanism === 'audio' || this.drivingMechanism === 'microphone' || this.drivingMechanism === 'desktop_audio') {
            this.actualAppliedFrequency = this.currentFrequency;
        } else {
             this.actualAppliedFrequency = this.currentFrequency;
        }
        
        if (this.uiElements.particleSpeedSlider) {
            const v = parseFloat(this.uiElements.particleSpeedSlider.value);
            const MAX_SIM_SPEED = 100.0; const MIN_SIM_SPEED = 0.1; const MID_SLIDER_VALUE = 50.0;
            if (v <= MID_SLIDER_VALUE) this.particleSimulationSpeedScale = MIN_SIM_SPEED + (1.0 - MIN_SIM_SPEED) * (v / MID_SLIDER_VALUE);
            else this.particleSimulationSpeedScale = 1.0 + (MAX_SIM_SPEED - 1.0) * ((v - MID_SLIDER_VALUE) / (100.0 - MID_SLIDER_VALUE));
            this.particleSimulationSpeedScale = Math.max(MIN_SIM_SPEED, Math.min(this.particleSimulationSpeedScale, MAX_SIM_SPEED));
            if (this.uiElements.speedValueText) this.uiElements.speedValueText.textContent = `${this.particleSimulationSpeedScale.toFixed(2)}x`;
        }

        if (isNaN(this.actualAppliedFrequency) || this.actualAppliedFrequency <= 0 || !isFinite(this.actualAppliedFrequency)) {
            this.actualAppliedFrequency = 20;
        }
        this._updateFrequencyControlsUI();
    }

    _initializeFDM_CPU_State() {
        this.currentGridSize = this._getOptimalGridSizeForFrequency(this.actualAppliedFrequency);
    }
    
    _initializeFDMArraysAndBuffers() {
        this.currentGridSize = this._roundToOddInteger(this._getOptimalGridSizeForFrequency(this.actualAppliedFrequency));
        this.plateWidth = this.PLATE_RADIUS * 2.0;
        this.plateHeight = this.PLATE_RADIUS * 2.0;
        this.dx = this.plateWidth / (this.currentGridSize - 1);
        this.dy = this.plateHeight / (this.currentGridSize - 1);
        this._updatePhysicalConstants();
        
        const freqFactor = Math.max(1.0, this.actualAppliedFrequency / 1000.0);

        if (this.D_FLEXURAL_RIGIDITY <= 0 || this.RHO_H_PLATE_SPECIFIC_DENSITY <= 0 || !isFinite(this.D_FLEXURAL_RIGIDITY) || !isFinite(this.RHO_H_PLATE_SPECIFIC_DENSITY)) {
            this.dt_simulation_step = 1e-7;
        } else {
            this.dt_simulation_step = (this.FDM_STABILITY_FACTOR * Math.pow(Math.min(this.dx, this.dy), 2) * Math.sqrt(this.RHO_H_PLATE_SPECIFIC_DENSITY / this.D_FLEXURAL_RIGIDITY)) / freqFactor;
        }
        
        if (this.dt_simulation_step <= 0 || isNaN(this.dt_simulation_step) || !isFinite(this.dt_simulation_step)) {
            this.dt_simulation_step = 1e-7;
        }
        
        this.u_curr_cpu_array2D = Array.from({ length: this.currentGridSize }, () => new Float32Array(this.currentGridSize).fill(0.0));
        this.u_prev_cpu_array2D = Array.from({ length: this.currentGridSize }, () => new Float32Array(this.currentGridSize).fill(0.0));
        this.u_next_temp_cpu_array2D = Array.from({ length: this.currentGridSize }, () => new Float32Array(this.currentGridSize).fill(0.0));
        this._initializeWaveFieldOnCPUArrays(this.u_curr_cpu_array2D, this.u_prev_cpu_array2D);
        this._updateModalExcitationPatternCPU();
        this.fdmConfiguredFrequency = this.actualAppliedFrequency;
    }

    _initializeWaveFieldOnCPUArrays(u_curr, u_prev) {
        const ampScale = this.PLATE_THICKNESS * 0.001;
        for (let i = 0; i < this.currentGridSize; i++) { u_curr[i].fill(0.0); u_prev[i].fill(0.0); }
        if (this.drivingMechanism === 'modal') {
            const b_zero = this._getBesselZero(this.mParameter, this.nParameter);
            if (b_zero === null || b_zero <= 0) { return; }
            const k_mn = b_zero / this.PLATE_RADIUS;
            for (let i = 0; i < this.currentGridSize; i++) {
                for (let j = 0; j < this.currentGridSize; j++) {
                    const x = (j / (this.currentGridSize - 1.0) - 0.5) * this.plateWidth;
                    const y = (i / (this.currentGridSize - 1.0) - 0.5) * this.plateHeight;
                    let disp = 0.0;
                    const r_phys = Math.hypot(x, y);
                    if (r_phys <= this.PLATE_RADIUS + this.dx * 0.15) {
                        const theta_phys = Math.atan2(y, x);
                        disp = this._besselJ_lib(this.mParameter, k_mn * r_phys) * Math.cos(this.mParameter * theta_phys);
                    }
                    u_curr[i][j] = disp * ampScale;
                    u_prev[i][j] = disp * ampScale;
                }
            }
        }
    }

    _updateModalExcitationPatternCPU() {
        this.modalExcitationPattern_cpu_array2D = Array.from({ length: this.currentGridSize }, () => new Float32Array(this.currentGridSize).fill(0.0));
        const b_zero = this._getBesselZero(this.mParameter, this.nParameter);
        if (b_zero !== null && b_zero > 0) {
            const k_mn = b_zero / this.PLATE_RADIUS;
            for (let i = 0; i < this.currentGridSize; i++) {
                for (let j = 0; j < this.currentGridSize; j++) {
                    const x = (j / (this.currentGridSize - 1.0) - 0.5) * this.plateWidth;
                    const y = (i / (this.currentGridSize - 1.0) - 0.5) * this.plateHeight;
                    const r_phys = Math.hypot(x, y);
                    if (r_phys <= this.PLATE_RADIUS + this.dx * 0.15) {
                        this.modalExcitationPattern_cpu_array2D[i][j] = this._besselJ_lib(this.mParameter, k_mn * r_phys);
                    } else this.modalExcitationPattern_cpu_array2D[i][j] = 0.0;
                }
            }
        }
    }
    
    _updatePlateFDM_CPU(deltaTime) {
        if (!this.u_curr_cpu_array2D || !this.u_prev_cpu_array2D || !this.u_next_temp_cpu_array2D) return;
        if (this.D_FLEXURAL_RIGIDITY <= 0 || this.RHO_H_PLATE_SPECIFIC_DENSITY <= 0 || !isFinite(this.D_FLEXURAL_RIGIDITY) || !isFinite(this.RHO_H_PLATE_SPECIFIC_DENSITY)) {
            return;
        }
        const scaledDeltaTime = deltaTime * this.particleSimulationSpeedScale;
        let numSteps = Math.ceil(scaledDeltaTime / this.dt_simulation_step);
        
        let maxSteps = this.BASE_MAX_FDM_STEPS_PER_FRAME;
        if (this.actualAppliedFrequency < 100) maxSteps = Math.min(this.BASE_MAX_FDM_STEPS_PER_FRAME * 2, Math.floor(this.BASE_MAX_FDM_STEPS_PER_FRAME * (this.currentGridSize < 61 ? 2.5 : 2.0)));
        else if (this.actualAppliedFrequency > 2500) maxSteps = Math.max(10, Math.floor(this.BASE_MAX_FDM_STEPS_PER_FRAME / 2));
        
        if (this.enableFDMOptimization && navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4) {
            maxSteps = Math.min(maxSteps, Math.max(10, Math.floor(this.BASE_MAX_FDM_STEPS_PER_FRAME / 1.5)));
        }

        numSteps = Math.min(numSteps, maxSteps) || 1;

        if (isNaN(this.dt_simulation_step) || this.dt_simulation_step <= 0 || !isFinite(this.dt_simulation_step)) { return; }
        let excAmp = this._getFrequencyDependentExcitationAmplitude(this.actualAppliedFrequency);
        const K_coeff = (this.dt_simulation_step ** 2 * this.D_FLEXURAL_RIGIDITY) / this.RHO_H_PLATE_SPECIFIC_DENSITY;
        const F_coeff = (this.dt_simulation_step ** 2) / this.RHO_H_PLATE_SPECIFIC_DENSITY;
        if (!isFinite(K_coeff) || !isFinite(F_coeff)) { return; }
        const gs = this.currentGridSize; const pr = this.PLATE_RADIUS; const pd = this.plateWidth;
        const mParam = Number(this.mParameter);
        
        const freqDampingCoefficient = 0.05;
        const baseDamping = this.FDM_DAMPING_FACTOR;
        const extraDamping = baseDamping * freqDampingCoefficient * Math.min(5.0, Math.max(0, (this.actualAppliedFrequency / 1000.0) - 1.0));
        const damp = baseDamping + extraDamping;

        const freq = this.actualAppliedFrequency;
        const excMode = (this.drivingMechanism === 'modal') ? 0 : 1;
        
        const inv_dx4 = (this.dx > 1e-9) ? (1.0 / (this.dx ** 4)) : 0;
        if (inv_dx4 === 0) { return; }
        if (this.uiElements.simulationProgress) this.uiElements.simulationProgress.value = 0;
        for (let step = 0; step < numSteps; step++) {
            if (this.uiElements.simulationProgress && step % Math.max(1, Math.floor(numSteps / 10)) === 0) this.uiElements.simulationProgress.value = ((step + 1) / numSteps) * 100;
            const s_time = this.simulationTime + step * this.dt_simulation_step;
            for (let i = 0; i < gs; i++) {
                for (let j = 0; j < gs; j++) {
                    const normX = (j / (gs - 1.0)) - 0.5; const normY = (i / (gs - 1.0)) - 0.5;
                    const physX = normX * pd; const physY = normY * pd;
                    if (physX ** 2 + physY ** 2 > pr ** 2 + this.dx ** 2 * 0.25) { this.u_next_temp_cpu_array2D[i][j] = 0.0; continue; }
                    const u_ij = this.u_curr_cpu_array2D[i][j];
                    const u_ip1j = (i + 1 < gs) ? this.u_curr_cpu_array2D[i + 1][j] : 0;
                    const u_im1j = (i - 1 >= 0) ? this.u_curr_cpu_array2D[i - 1][j] : 0;
                    const u_ijp1 = (j + 1 < gs) ? this.u_curr_cpu_array2D[i][j + 1] : 0;
                    const u_ijm1 = (j - 1 >= 0) ? this.u_curr_cpu_array2D[i][j - 1] : 0;
                    const u_ip1jp1 = (i + 1 < gs && j + 1 < gs) ? this.u_curr_cpu_array2D[i + 1][j + 1] : 0;
                    const u_ip1jm1 = (i + 1 < gs && j - 1 >= 0) ? this.u_curr_cpu_array2D[i + 1][j - 1] : 0;
                    const u_im1jp1 = (i - 1 >= 0 && j + 1 < gs) ? this.u_curr_cpu_array2D[i - 1][j + 1] : 0;
                    const u_im1jm1 = (i - 1 >= 0 && j - 1 >= 0) ? this.u_curr_cpu_array2D[i - 1][j - 1] : 0;
                    const u_ip2j = (i + 2 < gs) ? this.u_curr_cpu_array2D[i + 2][j] : 0;
                    const u_im2j = (i - 2 >= 0) ? this.u_curr_cpu_array2D[i - 2][j] : 0;
                    const u_ijp2 = (j + 2 < gs) ? this.u_curr_cpu_array2D[i][j + 2] : 0;
                    const u_ijm2 = (j - 2 >= 0) ? this.u_curr_cpu_array2D[i][j - 2] : 0;
                    const biharmonic = (20 * u_ij - 8 * (u_ip1j + u_im1j + u_ijp1 + u_ijm1) + 2 * (u_ip1jp1 + u_ip1jm1 + u_im1jp1 + u_im1jm1) + (u_ip2j + u_im2j + u_ijp2 + u_ijm2)) * inv_dx4;
                    let excForce = 0.0;
                    const timeSine = Math.sin(2 * Math.PI * freq * s_time);
                    if (excMode === 0 && this.modalExcitationPattern_cpu_array2D) {
                        const theta = Math.atan2(physY, physX);
                        excForce = excAmp * timeSine * this.modalExcitationPattern_cpu_array2D[i][j] * Math.cos(mParam * theta);
                    } else {
                        const cI = Math.floor(gs / 2); const cJ = Math.floor(gs / 2);
                        const distSq = (i - cI) ** 2 + (j - cJ) ** 2;
                        const excRadSq = Math.max(1.0, (gs * 0.04) ** 2);
                        if (distSq <= excRadSq) excForce = excAmp * timeSine * Math.exp(-distSq / (excRadSq * 0.5 + 1e-9));
                    }
                    let u_next = (2 * u_ij - this.u_prev_cpu_array2D[i][j]) - K_coeff * biharmonic + F_coeff * excForce;
                    this.u_next_temp_cpu_array2D[i][j] = isFinite(u_next) ? u_next * (1.0 - damp) : 0.0;
                }
            }
            let temp_ptr = this.u_prev_cpu_array2D;
            this.u_prev_cpu_array2D = this.u_curr_cpu_array2D;
            this.u_curr_cpu_array2D = this.u_next_temp_cpu_array2D;
            this.u_next_temp_cpu_array2D = temp_ptr;
        }
        this.simulationTime += numSteps * this.dt_simulation_step;
        if (this.uiElements.simulationProgress) this.uiElements.simulationProgress.value = 100;
    }

    _initializeFDM_GPU_State() {
        if (!this.gpuComputeManager || !this.useGPUComputing) {
            this._initializeFDM_CPU_State();
            return;
        }
        
        this.currentGridSize = this._getOptimalGridSizeForFrequency(this.actualAppliedFrequency);
    }
    
    _initializeFDM_GPU_ArraysAndBuffers() {
        if (!this.gpuComputeManager || !this.useGPUComputing) {
            this._initializeFDMArraysAndBuffers();
            return;
        }
        
        this.currentGridSize = this._roundToOddInteger(this._getOptimalGridSizeForFrequency(this.actualAppliedFrequency));
        this.plateWidth = this.PLATE_RADIUS * 2.0;
        this.plateHeight = this.PLATE_RADIUS * 2.0;
        this.dx = this.plateWidth / (this.currentGridSize - 1);
        this.dy = this.plateHeight / (this.currentGridSize - 1);
        this._updatePhysicalConstants();
        
        const freqFactor = Math.max(1.0, this.actualAppliedFrequency / 1000.0);

        if (this.D_FLEXURAL_RIGIDITY <= 0 || this.RHO_H_PLATE_SPECIFIC_DENSITY <= 0 || !isFinite(this.D_FLEXURAL_RIGIDITY) || !isFinite(this.RHO_H_PLATE_SPECIFIC_DENSITY)) {
            this.dt_simulation_step = 1e-7;
        } else {
            this.dt_simulation_step = (this.FDM_STABILITY_FACTOR * Math.pow(Math.min(this.dx, this.dy), 2) * Math.sqrt(this.RHO_H_PLATE_SPECIFIC_DENSITY / this.D_FLEXURAL_RIGIDITY)) / freqFactor;
        }
        
        if (this.dt_simulation_step <= 0 || isNaN(this.dt_simulation_step) || !isFinite(this.dt_simulation_step)) {
            this.dt_simulation_step = 1e-7;
        }
        
        // Создание GPU буферов
        try {
            this.gpuComputeManager.createFDMBuffers(this.currentGridSize);
            
            // Обновление uniform переменных для инициализации
            const initUniforms = [
                this.currentGridSize,
                this.PLATE_RADIUS,
                this.plateWidth,
                this.plateHeight,
                this.dx,
                this.dy,
                this.mParameter,
                this.nParameter,
                this.drivingMechanism === 'modal' ? 0 : 1,
                this.PLATE_THICKNESS * 0.001
            ];
            this.gpuComputeManager.updateInitUniforms(initUniforms);
            
            // Запуск инициализации поля на GPU
            this.gpuComputeManager.runInitFieldCompute(this.currentGridSize);
            
            this.fdmConfiguredFrequency = this.actualAppliedFrequency;
        } catch (error) {
            console.log('GPU FDM initialization: Compute shader not available, falling back to CPU');
            this.useGPUComputing = false;
            this._initializeFDMArraysAndBuffers();
        }
    }
    
    _updatePlateFDM_GPU(deltaTime) {
        if (!this.gpuComputeManager || !this.useGPUComputing) {
            this._updatePlateFDM_CPU(deltaTime);
            return;
        }
        
        if (this.D_FLEXURAL_RIGIDITY <= 0 || this.RHO_H_PLATE_SPECIFIC_DENSITY <= 0 || !isFinite(this.D_FLEXURAL_RIGIDITY) || !isFinite(this.RHO_H_PLATE_SPECIFIC_DENSITY)) {
            return;
        }
        
        const scaledDeltaTime = deltaTime * this.particleSimulationSpeedScale;
        let numSteps = Math.ceil(scaledDeltaTime / this.dt_simulation_step);
        
        let maxSteps = this.BASE_MAX_FDM_STEPS_PER_FRAME;
        if (this.actualAppliedFrequency < 100) maxSteps = Math.min(this.BASE_MAX_FDM_STEPS_PER_FRAME * 2, Math.floor(this.BASE_MAX_FDM_STEPS_PER_FRAME * (this.currentGridSize < 61 ? 2.5 : 2.0)));
        else if (this.actualAppliedFrequency > 2500) maxSteps = Math.max(10, Math.floor(this.BASE_MAX_FDM_STEPS_PER_FRAME / 2));
        
        if (this.enableFDMOptimization && navigator.hardwareConcurrency && navigator.hardwareConcurrency <= 4) {
            maxSteps = Math.min(maxSteps, Math.max(10, Math.floor(this.BASE_MAX_FDM_STEPS_PER_FRAME / 1.5)));
        }

        numSteps = Math.min(numSteps, maxSteps) || 1;

        if (isNaN(this.dt_simulation_step) || this.dt_simulation_step <= 0 || !isFinite(this.dt_simulation_step)) { return; }
        
        let excAmp = this._getFrequencyDependentExcitationAmplitude(this.actualAppliedFrequency);
        const K_coeff = (this.dt_simulation_step ** 2 * this.D_FLEXURAL_RIGIDITY) / this.RHO_H_PLATE_SPECIFIC_DENSITY;
        const F_coeff = (this.dt_simulation_step ** 2) / this.RHO_H_PLATE_SPECIFIC_DENSITY;
        if (!isFinite(K_coeff) || !isFinite(F_coeff)) { return; }
        
        const freqDampingCoefficient = 0.05;
        const baseDamping = this.FDM_DAMPING_FACTOR;
        const extraDamping = baseDamping * freqDampingCoefficient * Math.min(5.0, Math.max(0, (this.actualAppliedFrequency / 1000.0) - 1.0));
        const damp = baseDamping + extraDamping;

        const freq = this.actualAppliedFrequency;
        const excMode = (this.drivingMechanism === 'modal') ? 0 : 1;
        
        // Обновление uniform переменных для FDM
        const fdmUniforms = [
            this.currentGridSize,
            this.dt_simulation_step,
            this.dx,
            this.dy,
            K_coeff,
            F_coeff,
            damp,
            excAmp,
            freq,
            this.simulationTime,
            excMode,
            this.mParameter,
            this.PLATE_RADIUS,
            this.plateWidth,
            this.plateHeight
        ];
        try {
            this.gpuComputeManager.updateFDMUniforms(fdmUniforms);
            this.gpuComputeManager.runFDMCompute(this.currentGridSize, numSteps);
            this.simulationTime += numSteps * this.dt_simulation_step;
            if (this.uiElements.simulationProgress) this.uiElements.simulationProgress.value = 100;
        } catch (error) {
            console.log('GPU FDM: Compute shader not available, falling back to CPU');
            this.useGPUComputing = false;
            this._updatePlateFDM_CPU(deltaTime);
        }
    }

    _setupThreeJSScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.camera.position.set(0, this.PLATE_RADIUS * 1.6, this.PLATE_RADIUS * 2.0);
        this.camera.lookAt(0, 0, 0);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.outputColorSpace = THREE.SRGBColorSpace;
        
        this.renderer.shadowMap.enabled = this.enableShadows;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        const container = document.getElementById('scene-container');
        if (container) container.appendChild(this.renderer.domElement);
        this.orbitControls = new OrbitControls(this.camera, this.renderer.domElement);
        this.orbitControls.enableDamping = true; this.orbitControls.dampingFactor = 0.04;
        this.orbitControls.minDistance = this.PLATE_RADIUS * 0.3; this.orbitControls.maxDistance = this.PLATE_RADIUS * 12;
        this.orbitControls.target.set(0, 0, 0); this.orbitControls.autoRotate = false; this.orbitControls.enablePan = false;
        
        this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
        
        this.dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        this.dirLight1.position.set(this.PLATE_RADIUS * 2, this.PLATE_RADIUS * 3, this.PLATE_RADIUS * 1.5);
        this.dirLight1.castShadow = this.enableShadows;
        this.dirLight1.shadow.mapSize.width = 1024;
        this.dirLight1.shadow.mapSize.height = 1024;
        this.dirLight1.shadow.camera.near = 0.5;
        this.dirLight1.shadow.camera.far = this.PLATE_RADIUS * 10;
        this.dirLight1.shadow.camera.left = -this.PLATE_RADIUS * 2;
        this.dirLight1.shadow.camera.right = this.PLATE_RADIUS * 2;
        this.dirLight1.shadow.camera.top = this.PLATE_RADIUS * 2;
        this.dirLight1.shadow.camera.bottom = -this.PLATE_RADIUS * 2;
        this.scene.add(this.dirLight1);
        
        this.dirLight2 = new THREE.DirectionalLight(0xffddaa, 0.5);
        this.dirLight2.position.set(-this.PLATE_RADIUS * 2, this.PLATE_RADIUS * 1, -this.PLATE_RADIUS * 1.5);
        this.scene.add(this.dirLight2);

        const groundGeometry = new THREE.PlaneGeometry(this.PLATE_RADIUS * 4, this.PLATE_RADIUS * 4);
        const groundMaterial = new THREE.ShadowMaterial({ opacity: 0.3 });
        this.groundPlane = new THREE.Mesh(groundGeometry, groundMaterial);
        this.groundPlane.rotation.x = -Math.PI / 2;
        this.groundPlane.position.y = -this.PLATE_THICKNESS / 2 - 0.05;
        this.groundPlane.receiveShadow = this.enableShadows;
        this.groundPlane.visible = this.enableShadows;
        this.scene.add(this.groundPlane);
        
        // Инициализация GPU Compute Manager
        this._initializeGPUComputeManager();
    }

    async _initializeGPUComputeManager() {
        try {
            // Попытка использовать оптимизированный менеджер
            this.gpuComputeManager = new GPUComputeManagerOptimized(this.renderer);
            await this.gpuComputeManager.initialize();
            console.log('GPU Compute Manager (Optimized) initialized successfully');
            
            // Установка режима производительности на основе FPS
            this.gpuComputeManager.setPerformanceMode('balanced');
        } catch (error) {
            console.log('GPU Compute Manager (Optimized) failed, trying standard version');
            try {
                this.gpuComputeManager = new GPUComputeManager(this.renderer);
                await this.gpuComputeManager.initialize();
                console.log('GPU Compute Manager (Standard) initialized successfully');
            } catch (fallbackError) {
                console.log('GPU Compute Manager: Compute shader extension not supported, falling back to CPU');
                this.useGPUComputing = false;
                this.gpuComputeManager = null;
            }
        }
    }

    _createParticleSystem() {
        if (this.particlesMesh) {
            this.scene.remove(this.particlesMesh);
            this.particlesMesh.geometry.dispose();
            this.particlesMesh.material.dispose();
            this.particlesMesh = null;
        }
        
        const effectiveParticleCount = Math.round(this.PARTICLE_COUNT);
        if (effectiveParticleCount <= 0) return;

        const geom = new THREE.SphereGeometry(this.PARTICLE_SIZE, 5, 3);
        const mat = new THREE.ShaderMaterial({
            uniforms: { globalColor: { value: new THREE.Color(0x00ddff) } },
            vertexShader: `void main() { vec4 worldPos = instanceMatrix * vec4(position, 1.0); gl_Position = projectionMatrix * modelViewMatrix * worldPos; }`,
            fragmentShader: `uniform vec3 globalColor; void main() { gl_FragColor = vec4(globalColor, 1.0); }`,
        });
        this.particlesMesh = new THREE.InstancedMesh(geom, mat, effectiveParticleCount);
        this.particlesMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        this.particlesMesh.castShadow = this.enableShadows;
        this.scene.add(this.particlesMesh);
        
        this.particleData.length = 0;
        for (let i = 0; i < effectiveParticleCount; i++) {
            this.particleData.push({
                position: new THREE.Vector3(),
                velocity: new THREE.Vector3(),
                isHidden: false,
                stuckTimer: 0
            });
        }
        this._resetParticlePositionsAndVelocities();
    }

    _resetParticlePositionsAndVelocities() {
        const effectiveParticleCount = Math.round(this.PARTICLE_COUNT);
        if (!this.particlesMesh || this.particleData.length !== effectiveParticleCount || effectiveParticleCount === 0) return;

        const tempMat = new THREE.Matrix4();
        for (let i = 0; i < effectiveParticleCount; i++) {
            const r = Math.sqrt(Math.random()) * this.PLATE_RADIUS;
            const angle = Math.random() * 2 * Math.PI;
            const x = r * Math.cos(angle);
            const y = r * Math.sin(angle);
            if (this.particleData[i]) {
                this.particleData[i].position.set(x, y, 0);
                this.particleData[i].velocity.set(0, 0, 0);
                this.particleData[i].isHidden = false;
                this.particleData[i].stuckTimer = 0;
            } else {
                this.particleData[i] = {
                    position: new THREE.Vector3(x, y, 0),
                    velocity: new THREE.Vector3(0, 0, 0),
                    isHidden: false,
                    stuckTimer: 0
                };
            }
            tempMat.identity().setPosition(x, 0, y);
            this.particlesMesh.setMatrixAt(i, tempMat);
        }
        if (this.particlesMesh) this.particlesMesh.instanceMatrix.needsUpdate = true;
    }

    _getDisplacementFromCPUField(field, r_float, c_float) { 
        if (!field || field.length !== this.currentGridSize || !field[0] || field[0].length !== this.currentGridSize) return 0; 
        const i0 = Math.floor(r_float); 
        const j0 = Math.floor(c_float); 
        const i1 = i0 + 1; 
        const j1 = j0 + 1; 
        if (i0 < 0 || i1 >= this.currentGridSize || j0 < 0 || j1 >= this.currentGridSize) { 
            const cl_i = Math.max(0, Math.min(Math.round(r_float), this.currentGridSize - 1)); 
            const cl_j = Math.max(0, Math.min(Math.round(c_float), this.currentGridSize - 1)); 
            return field[cl_i][cl_j]; 
        } 
        const tx = c_float - j0; 
        const ty = r_float - i0; 
        const v00 = field[i0][j0]; 
        const v01 = field[i0][j1]; 
        const v10 = field[i1][j0]; 
        const v11 = field[i1][j1]; 
        const r1 = (1 - tx) * v00 + tx * v01; 
        const r2 = (1 - tx) * v10 + tx * v11; 
        return (1 - ty) * r1 + ty * r2; 
    }
    
    _getGradientFromCPUField(field, r_int, c_int) { 
        const ri = Math.max(1, Math.min(r_int, this.currentGridSize - 2)); 
        const ci = Math.max(1, Math.min(c_int, this.currentGridSize - 2)); 
        if (!field || ri <= 0 || ri >= this.currentGridSize - 1 || ci <= 0 || ci >= this.currentGridSize - 1 || !isFinite(this.dx) || !isFinite(this.dy) || this.dx === 0 || this.dy === 0) return { gradX: 0, gradY: 0 }; 
        const vpr = field[ri + 1][ci]; 
        const vmr = field[ri - 1][ci]; 
        const vpc = field[ri][ci + 1]; 
        const vmc = field[ri][ci - 1]; 
        const gy = (vpr - vmr) / (2 * this.dy); 
        const gx = (vpc - vmc) / (2 * this.dx); 
        return { gradX: isFinite(gx) ? gx : 0, gradY: isFinite(gy) ? gy : 0 }; 
    }

    _getDisplacementFromGPUField(r_float, c_float) {
        if (!this.gpuComputeManager || !this.useGPUComputing) {
            return this._getDisplacementFromCPUField(this.u_curr_cpu_array2D, r_float, c_float);
        }
        
        try {
            const fdmData = this.gpuComputeManager.getFDMData(this.currentGridSize);
            const i0 = Math.floor(r_float);
            const j0 = Math.floor(c_float);
            const i1 = i0 + 1;
            const j1 = j0 + 1;
            
            if (i0 < 0 || i1 >= this.currentGridSize || j0 < 0 || j1 >= this.currentGridSize) {
                const cl_i = Math.max(0, Math.min(Math.round(r_float), this.currentGridSize - 1));
                const cl_j = Math.max(0, Math.min(Math.round(c_float), this.currentGridSize - 1));
                return fdmData[cl_i * this.currentGridSize + cl_j] || 0.0;
            }
            
            const tx = c_float - j0;
            const ty = r_float - i0;
            const v00 = fdmData[i0 * this.currentGridSize + j0] || 0.0;
            const v01 = fdmData[i0 * this.currentGridSize + j1] || 0.0;
            const v10 = fdmData[i1 * this.currentGridSize + j0] || 0.0;
            const v11 = fdmData[i1 * this.currentGridSize + j1] || 0.0;
            
            const r1 = (1 - tx) * v00 + tx * v01;
            const r2 = (1 - tx) * v10 + tx * v11;
            return (1 - ty) * r1 + ty * r2;
        } catch (error) {
            console.log('GPU displacement: Compute shader not available, falling back to CPU');
            this.useGPUComputing = false;
            return this._getDisplacementFromCPUField(this.u_curr_cpu_array2D, r_float, c_float);
        }
    }

    _getGradientFromGPUField(r_int, c_int) {
        if (!this.gpuComputeManager || !this.useGPUComputing) {
            return this._getGradientFromCPUField(this.u_curr_cpu_array2D, r_int, c_int);
        }
        
        try {
            const fdmData = this.gpuComputeManager.getFDMData(this.currentGridSize);
            const ri = Math.max(1, Math.min(r_int, this.currentGridSize - 2));
            const ci = Math.max(1, Math.min(c_int, this.currentGridSize - 2));
            
            if (ri <= 0 || ri >= this.currentGridSize - 1 || ci <= 0 || ci >= this.currentGridSize - 1 || !isFinite(this.dx) || !isFinite(this.dy) || this.dx === 0 || this.dy === 0) {
                return { gradX: 0, gradY: 0 };
            }
            
            const vpr = fdmData[(ri + 1) * this.currentGridSize + ci] || 0.0;
            const vmr = fdmData[(ri - 1) * this.currentGridSize + ci] || 0.0;
            const vpc = fdmData[ri * this.currentGridSize + (ci + 1)] || 0.0;
            const vmc = fdmData[ri * this.currentGridSize + (ci - 1)] || 0.0;
            
            const gy = (vpr - vmr) / (2 * this.dy);
            const gx = (vpc - vmc) / (2 * this.dx);
            
            return { gradX: isFinite(gx) ? gx : 0, gradY: isFinite(gy) ? gy : 0 };
        } catch (error) {
            console.log('GPU gradient: Compute shader not available, falling back to CPU');
            this.useGPUComputing = false;
            return this._getGradientFromCPUField(this.u_curr_cpu_array2D, r_int, c_int);
        }
    }
    
    _updateParticles(deltaTime) {
        const effectiveParticleCount = Math.round(this.PARTICLE_COUNT);
        if (!this.particlesMesh || this.particleData.length !== effectiveParticleCount || !this.u_curr_cpu_array2D || !this.dx || !this.dy || effectiveParticleCount === 0) return;

        const scaledDT = deltaTime * this.particleSimulationSpeedScale;
        
        let adaptDamp = this.PARTICLE_DAMPING_BASE;
        let forceMult = this.PARTICLE_EFFECTIVE_FORCE_MULTIPLIER_BASE;

        if (this.actualAppliedFrequency < 300) {
            adaptDamp = Math.min(0.97, this.PARTICLE_DAMPING_BASE + 0.03);
            forceMult = this.PARTICLE_EFFECTIVE_FORCE_MULTIPLIER_BASE * 1.25;
        }
        
        const hiddenParticlePosition = new THREE.Vector3(this.PLATE_RADIUS * 100, 0, 0);
        const stuckDispThreshold = this.MAX_VISUAL_AMPLITUDE * STUCK_PARTICLE_DISPLACEMENT_THRESHOLD_FACTOR;

        for (let idx = 0; idx < effectiveParticleCount; idx++) {
            const p = this.particleData[idx];
            if (p.isHidden && this.enableStuckParticleCulling) {
                this.particleInstanceMatrix.setPosition(hiddenParticlePosition.x, hiddenParticlePosition.y, hiddenParticlePosition.z);
                this.particlesMesh.setMatrixAt(idx, this.particleInstanceMatrix);
                continue;
            }
            p.isHidden = false;

            if (this.areParticlesFrozen) {
                const pos = p.position;
                let h = 0;
                if (this.u_curr_cpu_array2D) { const nX = (pos.x / this.plateWidth) + 0.5; const nY = (pos.y / this.plateHeight) + 0.5; const rF = nY * (this.currentGridSize - 1); const cF = nX * (this.currentGridSize - 1); const d = this._getDisplacementFromGPUField(rF, cF); h = THREE.MathUtils.clamp(d * this.VISUAL_DEFORMATION_SCALE, -this.MAX_VISUAL_AMPLITUDE, this.MAX_VISUAL_AMPLITUDE); }
                const rx = pos.x * Math.cos(this.plateRotationAngle) - pos.y * Math.sin(this.plateRotationAngle);
                const rz = pos.x * Math.sin(this.plateRotationAngle) + pos.y * Math.cos(this.plateRotationAngle);
                this.particleInstanceMatrix.setPosition(rx, h, rz);
                this.particlesMesh.setMatrixAt(idx, this.particleInstanceMatrix);
                continue;
            }

            const pos = p.position;
            const vel = p.velocity;
            const normGX = (pos.x / this.plateWidth) + 0.5;
            const normGY = (pos.y / this.plateHeight) + 0.5;
            const fdmR = normGY * (this.currentGridSize - 1);
            const fdmC = normGX * (this.currentGridSize - 1);
            const disp = this._getDisplacementFromGPUField(fdmR, fdmC);
            const grad = this._getGradientFromGPUField(Math.round(fdmR), Math.round(fdmC));
            let fx = -2 * disp * grad.gradX * forceMult;
            let fy = -2 * disp * grad.gradY * forceMult;
            if (this.ENABLE_PARTICLE_REPULSION) {
                let repX = 0, repY = 0;
                let nChecked = 0;
                const step = Math.max(1, Math.floor(effectiveParticleCount / (this.MAX_REPULSION_NEIGHBORS_CHECK * 3 + 1)));
                for (let otherIdx = (idx + Math.floor(Math.random() * step)) % effectiveParticleCount; nChecked < this.MAX_REPULSION_NEIGHBORS_CHECK; otherIdx = (otherIdx + step) % effectiveParticleCount) {
                    if (idx === otherIdx || otherIdx < 0 || otherIdx >= effectiveParticleCount) continue;
                    const otherP = this.particleData[otherIdx];
                    if (!otherP || otherP.isHidden) continue;
                    const dx_p = pos.x - otherP.position.x; const dy_p = pos.y - otherP.position.y;
                    const distSq = dx_p ** 2 + dy_p ** 2;
                    if (distSq < this.PARTICLE_REPULSION_RADIUS ** 2 && distSq > 1e-9) {
                        const dist_p = Math.sqrt(distSq);
                        const repMag = this.PARTICLE_REPULSION_STRENGTH * Math.pow(this.PARTICLE_REPULSION_RADIUS - dist_p, 2) / (dist_p + 1e-9);
                        repX += repMag * (dx_p / dist_p); repY += repMag * (dy_p / dist_p);
                    }
                    nChecked++;
                    if (otherIdx === (idx + Math.floor(Math.random() * step)) % effectiveParticleCount && nChecked > 5 && step > 0 && effectiveParticleCount > this.MAX_REPULSION_NEIGHBORS_CHECK * 1.5) break;
                } fx += repX; fy += repY;
            }
            const effDamp = Math.abs(disp) < 1e-4 ? 0.99 : adaptDamp;
            vel.x = (vel.x + fx * scaledDT) * effDamp;
            vel.y = (vel.y + fy * scaledDT) * effDamp;
            const speed = Math.hypot(vel.x, vel.y);

            if (this.enableStuckParticleCulling) {
                if (speed < STATIONARY_VELOCITY_THRESHOLD && Math.abs(disp) > stuckDispThreshold) {
                    p.stuckTimer += deltaTime;
                    if (p.stuckTimer > 0.5) {
                        p.isHidden = true;
                        p.velocity.set(0, 0, 0);
                        this.particleInstanceMatrix.setPosition(hiddenParticlePosition.x, hiddenParticlePosition.y, hiddenParticlePosition.z);
                        this.particlesMesh.setMatrixAt(idx, this.particleInstanceMatrix);
                        continue;
                    }
                } else {
                    p.stuckTimer = 0;
                }
            }

            if (speed > this.MAX_PARTICLE_SPEED) {
                vel.x *= this.MAX_PARTICLE_SPEED / speed; vel.y *= this.MAX_PARTICLE_SPEED / speed;
            }
            pos.x += vel.x * scaledDT;
            pos.y += vel.y * scaledDT;
            const radAfter = Math.hypot(pos.x, pos.y);
            if (radAfter > this.PLATE_RADIUS) {
                const corrRatio = this.PLATE_RADIUS / radAfter; pos.x *= corrRatio; pos.y *= corrRatio;
                const normBx = pos.x / this.PLATE_RADIUS; const normBy = pos.y / this.PLATE_RADIUS;
                const vDotN = vel.x * normBx + vel.y * normBy;
                if (vDotN > 0) {
                    vel.x -= (1 + this.PARTICLE_BOUNDARY_RESTITUTION) * vDotN * normBx;
                    vel.y -= (1 + this.PARTICLE_BOUNDARY_RESTITUTION) * vDotN * normBy;
                }
            }
            let visHeight = THREE.MathUtils.clamp(disp * this.VISUAL_DEFORMATION_SCALE, -this.MAX_VISUAL_AMPLITUDE, this.MAX_VISUAL_AMPLITUDE);
            const rotX = pos.x * Math.cos(this.plateRotationAngle) - pos.y * Math.sin(this.plateRotationAngle);
            const rotZ = pos.x * Math.sin(this.plateRotationAngle) + pos.y * Math.cos(this.plateRotationAngle);
            this.particleInstanceMatrix.setPosition(rotX, visHeight, rotZ);
            this.particlesMesh.setMatrixAt(idx, this.particleInstanceMatrix);
        }
        if (this.particlesMesh) this.particlesMesh.instanceMatrix.needsUpdate = true;
    }

    _updateDynamicParticleDensity() {
        if (!this.enableDynamicParticleDensity ||
            (this.drivingMechanism !== 'audio' && this.drivingMechanism !== 'microphone' && this.drivingMechanism !== 'desktop_audio') ||
            (!this.isAudioFilePlaying && !this.isMicrophoneEnabled && !this.isDesktopAudioEnabled) ||
            this.isAudioFilePaused ||
            !this.fftAnalyserNode ||
            !this.mainAudioContext ||
            this.mainAudioContext.state !== 'running') {
            return;
        }

        const currentTime = this.mainAudioContext.currentTime;
        if (currentTime - this.lastParticleCountUpdateTime < (PARTICLE_COUNT_UPDATE_THROTTLE_MS / 1000.0)) {
            return;
        }
        this.lastParticleCountUpdateTime = currentTime;

        this.fftAnalyserNode.getByteFrequencyData(this.frequencyData);
        let sumVol = 0;
        for (let i = 0; i < this.frequencyData.length; i++) {
            sumVol += this.frequencyData[i];
        }
        const avgVol = this.frequencyData.length > 0 ? sumVol / this.frequencyData.length : 0;
        
        const volumeFactor = THREE.MathUtils.clamp(avgVol / 128.0, 0.05, 1.5);

        let newParticleCount = MIN_DYNAMIC_PARTICLE_COUNT + (this.MAX_PARTICLE_COUNT_USER_SETTING - MIN_DYNAMIC_PARTICLE_COUNT) * volumeFactor;
        newParticleCount = Math.round(THREE.MathUtils.clamp(newParticleCount, MIN_DYNAMIC_PARTICLE_COUNT, this.MAX_PARTICLE_COUNT_USER_SETTING) / 100) * 100;

        if (Math.abs(newParticleCount - this.PARTICLE_COUNT) > this.PARTICLE_COUNT * 0.05 ||
            (newParticleCount > this.PARTICLE_COUNT && newParticleCount < this.MAX_PARTICLE_COUNT_USER_SETTING) ||
            (newParticleCount < this.PARTICLE_COUNT && newParticleCount > MIN_DYNAMIC_PARTICLE_COUNT)
            ) {
            this.PARTICLE_COUNT = newParticleCount;
            this._createParticleSystem();
        }
    }

    _updateParticles_GPU(deltaTime) {
        if (!this.gpuComputeManager || !this.useGPUComputing) {
            this._updateParticles(deltaTime);
            return;
        }
        
        const effectiveParticleCount = Math.round(this.PARTICLE_COUNT);
        if (!this.particlesMesh || this.particleData.length !== effectiveParticleCount || !this.dx || !this.dy || effectiveParticleCount === 0) return;

        // Создание буферов для частиц, если они еще не созданы
        if (!this.gpuComputeManager.particlePositionsBuffer) {
            this.gpuComputeManager.createParticleBuffers(effectiveParticleCount);
        }
        
        // Обновление данных частиц в GPU буферах
        const positions = new Float32Array(effectiveParticleCount * 4);
        const velocities = new Float32Array(effectiveParticleCount * 4);
        const states = new Int32Array(effectiveParticleCount);
        
        for (let i = 0; i < effectiveParticleCount; i++) {
            const p = this.particleData[i];
            positions[i * 4] = p.position.x;
            positions[i * 4 + 1] = p.position.y;
            positions[i * 4 + 2] = p.position.z;
            positions[i * 4 + 3] = 1.0;
            
            velocities[i * 4] = p.velocity.x;
            velocities[i * 4 + 1] = p.velocity.y;
            velocities[i * 4 + 2] = p.velocity.z;
            velocities[i * 4 + 3] = 1.0;
            
            states[i] = p.isHidden ? 1 : 0;
        }
        
        // Загрузка данных в GPU буферы
        this.gpuComputeManager.gl.bindBuffer(this.gpuComputeManager.gl.SHADER_STORAGE_BUFFER, this.gpuComputeManager.particlePositionsBuffer);
        this.gpuComputeManager.gl.bufferSubData(this.gpuComputeManager.gl.SHADER_STORAGE_BUFFER, 0, positions);
        
        this.gpuComputeManager.gl.bindBuffer(this.gpuComputeManager.gl.SHADER_STORAGE_BUFFER, this.gpuComputeManager.particleVelocitiesBuffer);
        this.gpuComputeManager.gl.bufferSubData(this.gpuComputeManager.gl.SHADER_STORAGE_BUFFER, 0, velocities);
        
        this.gpuComputeManager.gl.bindBuffer(this.gpuComputeManager.gl.SHADER_STORAGE_BUFFER, this.gpuComputeManager.particleStatesBuffer);
        this.gpuComputeManager.gl.bufferSubData(this.gpuComputeManager.gl.SHADER_STORAGE_BUFFER, 0, states);
        
        // Обновление uniform переменных для частиц
        const particleUniforms = [
            this.currentGridSize,
            effectiveParticleCount,
            deltaTime * this.particleSimulationSpeedScale,
            this.PLATE_RADIUS,
            this.plateWidth,
            this.plateHeight,
            this.dx,
            this.dy,
            this.PARTICLE_EFFECTIVE_FORCE_MULTIPLIER_BASE,
            this.PARTICLE_DAMPING_BASE,
            this.PARTICLE_REPULSION_RADIUS,
            this.PARTICLE_REPULSION_STRENGTH,
            this.MAX_PARTICLE_SPEED,
            this.PARTICLE_BOUNDARY_RESTITUTION,
            this.VISUAL_DEFORMATION_SCALE,
            this.MAX_VISUAL_AMPLITUDE,
            this.plateRotationAngle,
            this.ENABLE_PARTICLE_REPULSION ? 1 : 0,
            this.MAX_REPULSION_NEIGHBORS_CHECK,
            STATIONARY_VELOCITY_THRESHOLD,
            this.MAX_VISUAL_AMPLITUDE * STUCK_PARTICLE_DISPLACEMENT_THRESHOLD_FACTOR,
            this.enableStuckParticleCulling ? 1 : 0,
            this.PLATE_RADIUS * 100, 0, 0, // hiddenPosition
            this.actualAppliedFrequency // Добавляем частоту для адаптивного демпфирования
        ];
        try {
            this.gpuComputeManager.updateParticleUniforms(particleUniforms);
            this.gpuComputeManager.runParticleCompute(effectiveParticleCount);
            
            // Получение результатов обратно
            const result = this.gpuComputeManager.getParticleData(effectiveParticleCount);
            
            // Обновление данных частиц
            for (let i = 0; i < effectiveParticleCount; i++) {
                const p = this.particleData[i];
                p.position.set(result.positions[i * 4], result.positions[i * 4 + 1], result.positions[i * 4 + 2]);
                p.velocity.set(result.velocities[i * 4], result.velocities[i * 4 + 1], result.velocities[i * 4 + 2]);
                p.isHidden = result.states[i] === 1;
                
                // Обновление матрицы инстансов
                this.particleInstanceMatrix.setPosition(p.position.x, p.position.y, p.position.z);
                this.particlesMesh.setMatrixAt(i, this.particleInstanceMatrix);
            }
            
            if (this.particlesMesh) this.particlesMesh.instanceMatrix.needsUpdate = true;
        } catch (error) {
            console.log('GPU particles: Compute shader not available, falling back to CPU');
            this.useGPUComputing = false;
            this._updateParticles(deltaTime);
        }
    }
    
    _setupWebAudioSystem() {
        try {
            this.mainAudioContext = new (window.AudioContext || window.webkitAudioContext)();
            if (!this.mainAudioContext) throw new Error("AudioContext failed.");
            
            this.mainAudioContext.addEventListener('statechange', this._handleAudioContextStateChange.bind(this));

            const resume = async () => { if (this.mainAudioContext?.state === 'suspended') await this.mainAudioContext.resume(); document.removeEventListener('click', resume); document.removeEventListener('touchstart', resume); document.removeEventListener('keydown', resume); };
            document.addEventListener('click', resume, { once: true }); document.addEventListener('touchstart', resume, { once: true }); document.addEventListener('keydown', resume, { once: true });
            
            this.pitchDetectorAnalyserNode = this.mainAudioContext.createAnalyser();
            this.pitchDetectorAnalyserNode.fftSize = 2048;
            this.pitchDetectorAnalyserNode.smoothingTimeConstant = 0;
            this.pitchDetectorSignalBuffer = new Float32Array(this.pitchDetectorAnalyserNode.fftSize);
            
            const createBpmAnalyzer = () => {
                const analyser = this.mainAudioContext.createAnalyser();
                analyser.fftSize = 256;
                analyser.smoothingTimeConstant = 0.3;
                return analyser;
            };

            this.bpmAnalyzers.low.analyser = createBpmAnalyzer();
            this.bpmAnalyzers.mid.analyser = createBpmAnalyzer();
            this.bpmAnalyzers.high.analyser = createBpmAnalyzer();

            this.fftAnalyserNode = this.mainAudioContext.createAnalyser();
            this.fftAnalyserNode.fftSize = 256;
            this.frequencyData = new Uint8Array(this.fftAnalyserNode.frequencyBinCount);

            const sr = this.mainAudioContext.sampleRate;
            this.MIN_SAMPLES_FOR_PITCH_DETECTION = Math.max(4, Math.floor(sr / PITCH_MAX_FREQUENCY_HZ));
            this.MAX_SAMPLES_FOR_PITCH_DETECTION = Math.min(Math.floor(this.pitchDetectorAnalyserNode.fftSize / 2), Math.floor(sr / PITCH_MIN_FREQUENCY_HZ));
        } catch (e) {
            if (this.uiElements.toggleSoundButton) { this.uiElements.toggleSoundButton.textContent = 'Звук: Ошибка'; this.uiElements.toggleSoundButton.disabled = true; }
            if (this.uiElements.audioFileInput) this.uiElements.audioFileInput.disabled = true;
            if (this.uiElements.toggleMicrophoneButton) { this.uiElements.toggleMicrophoneButton.textContent = 'Микрофон: Ошибка'; this.uiElements.toggleMicrophoneButton.disabled = true; }
            if (this.uiElements.toggleDesktopAudioButton) { this.uiElements.toggleDesktopAudioButton.textContent = 'Перехват: Ошибка'; this.uiElements.toggleDesktopAudioButton.disabled = true; }
            alert("Ошибка: Web Audio API не доступен.");
        }
    }
    
    _handleAudioContextStateChange() {
        if (this.mainAudioContext.state === 'interrupted') {
        } else if (this.mainAudioContext.state === 'running') {
        }
    }

    _setupSubtitleSystem() {
        this.subtitleContainer = this.uiElements['subtitle-container'];
        if (!this.subtitleContainer) {
            return;
        }
        this.currentSubtitles = [];
        this.subtitleContainer.textContent = '';
        this.subtitleContainer.classList.remove('visible');
    }

    _getFrequencyDependentExcitationAmplitude(freq) { 
        const base = this.EXCITATION_FREQ_DEP_BASE_AMP; 
        if (freq <= 0 || !isFinite(freq)) return base * this.EXCITATION_FREQ_DEP_MAX_FACTOR; 
        if (freq < this.EXCITATION_FREQ_DEP_LOW_CUTOFF) return base * this.EXCITATION_FREQ_DEP_MAX_FACTOR; 
        else if (freq < this.EXCITATION_FREQ_DEP_HIGH_CUTOFF) { 
            const factorRange = this.EXCITATION_FREQ_DEP_MAX_FACTOR - this.EXCITATION_FREQ_DEP_MIN_FACTOR; 
            const freqRange = this.EXCITATION_FREQ_DEP_HIGH_CUTOFF - this.EXCITATION_FREQ_DEP_LOW_CUTOFF; 
            if (freqRange <= 0) return base * this.EXCITATION_FREQ_DEP_MAX_FACTOR; 
            return base * (this.EXCITATION_FREQ_DEP_MAX_FACTOR - factorRange * ((freq - this.EXCITATION_FREQ_DEP_LOW_CUTOFF) / freqRange)); 
        } else return base * this.EXCITATION_FREQ_DEP_MIN_FACTOR; 
    }

    _midiNoteNumberFromFrequency(freq) { return Math.round(12 * (Math.log(freq / 440) / Math.log(2))) + 69; }
    _frequencyFromMIDINoteNumber(midi) { return 440 * Math.pow(2, (midi - 69) / 12); }
    _centsOffFromTruePitch(freq, midi) { return Math.floor(1200 * Math.log(freq / this._frequencyFromMIDINoteNumber(midi)) / Math.log(2)); }
    
    _autoCorrelatePitch(buffer, sampleRate) {
        let bufSize = buffer.length;
        let rms = 0;
        for (let i = 0; i < bufSize; i++) {
            let v = buffer[i];
            rms += v * v;
        }
        rms = Math.sqrt(rms / bufSize);

        const noiseFloor = 0.005 + (rms * 0.1);
        if (rms < noiseFloor) return -1;

        let C = new Float32Array(bufSize);
        for (let lag = 0; lag < bufSize; lag++) {
            let sum = 0;
            for (let i = 0; i < bufSize - lag; i++) sum += buffer[i] * buffer[i + lag];
            C[lag] = sum;
        }

        let C_norm = new Float32Array(bufSize);
        let C0 = C[0] > 1e-9 ? C[0] : 1e-9;
        for (let lag = 0; lag < bufSize; lag++) {
            C_norm[lag] = C[lag] / C0;
        }
        
        let candidates = [];
        for (let off = this.MIN_SAMPLES_FOR_PITCH_DETECTION; off <= this.MAX_SAMPLES_FOR_PITCH_DETECTION && off < C_norm.length - 1; off++) {
            if (C_norm[off] > C_norm[off - 1] && C_norm[off] > C_norm[off + 1] && C_norm[off] > 0.1) {
                candidates.push({ lag: off, value: C_norm[off] });
            }
        }

        if (candidates.length === 0) return -1;
        candidates.sort((a, b) => b.value - a.value);
        candidates = candidates.slice(0, 3);

        let bestCandidate = null;
        let bestScore = -Infinity;

        for (let candidate of candidates) {
            let period_cand = candidate.lag;
            let freq_cand = sampleRate / period_cand;
            if (freq_cand < PITCH_MIN_FREQUENCY_HZ || freq_cand > PITCH_MAX_FREQUENCY_HZ) continue;

            let score = candidate.value * (1 + (rms - noiseFloor) * 0.5);

            if (this.lastStablePitchFrequency > 0) {
                let freqRatio = freq_cand / this.lastStablePitchFrequency;
                if (Math.abs(freqRatio - 1) < 0.25) score += 0.5;
                else if (Math.abs(freqRatio - 0.5) < 0.1 || Math.abs(freqRatio - 2) < 0.2) score += 0.3;
            }

            let harmonicPresenceScore = 0;
            for (let harmonicMultiple = 2; harmonicMultiple <= 3; harmonicMultiple++) {
                let harmonicPeriod = Math.round(period_cand / harmonicMultiple);
                if (harmonicPeriod >= this.MIN_SAMPLES_FOR_PITCH_DETECTION && harmonicPeriod < C_norm.length) {
                    if (C_norm[harmonicPeriod] > 0.05) {
                        harmonicPresenceScore += C_norm[harmonicPeriod] * 0.2;
                    }
                }
            }
            score += harmonicPresenceScore;

            if (score > bestScore) {
                bestScore = score;
                bestCandidate = candidate;
            }
        }

        if (!bestCandidate) return -1;

        let period = parseFloat(bestCandidate.lag);
        if (bestCandidate.lag > 0 && bestCandidate.lag < C.length - 1) {
            let y1 = C[bestCandidate.lag - 1], y2 = C[bestCandidate.lag], y3 = C[bestCandidate.lag + 1];
            let denominator = 2 * (2 * y2 - y1 - y3);
            if (Math.abs(denominator) > 1e-6) {
                let ps = (y3 - y1) / denominator;
                if (isFinite(ps) && Math.abs(ps) < 0.5) period = bestCandidate.lag + ps;
            }
        }
        
        let finalFreq = period > 0 ? sampleRate / period : -1;
        
        if (!(finalFreq >= PITCH_MIN_FREQUENCY_HZ && finalFreq <= PITCH_MAX_FREQUENCY_HZ)) {
            const rawBestFreq = sampleRate / bestCandidate.lag;
            if (rawBestFreq >= PITCH_MIN_FREQUENCY_HZ && rawBestFreq <= PITCH_MAX_FREQUENCY_HZ) {
                finalFreq = rawBestFreq;
            } else {
                finalFreq = -1;
            }
        }
        return finalFreq;
    }

    async _toggleGeneratedSoundPlayback() {
        this.isGeneratedSoundEnabled = !this.isGeneratedSoundEnabled;
        if (this.generatedSoundOscillator) {
            try { this.generatedSoundOscillator.stop(); this.generatedSoundOscillator.disconnect(); } catch (e) { }
            this.generatedSoundOscillator = null;
            if (this.generatedSoundGainNode) { try { this.generatedSoundGainNode.disconnect(); } catch (e) { } this.generatedSoundGainNode = null; }
        }
        if (this.isGeneratedSoundEnabled) {
            if (!this.mainAudioContext) { this._setupWebAudioSystem(); if (!this.mainAudioContext) { this.isGeneratedSoundEnabled = false; this._updateUIToggleButtons(); return; } }
            if (this.mainAudioContext.state === 'suspended') await this.mainAudioContext.resume();
            
            if (this.isAudioFilePlaying || this.isAudioFilePaused) await this._stopLoadedAudioFilePlayback(true);
            if (this.isMicrophoneEnabled) await this._toggleMicrophoneInput();
            if (this.isDesktopAudioEnabled) await this._toggleDesktopAudio();

            this.generatedSoundOscillator = this.mainAudioContext.createOscillator();
            this.generatedSoundGainNode = this.mainAudioContext.createGain();
            this.generatedSoundOscillator.connect(this.generatedSoundGainNode);
            this.generatedSoundGainNode.connect(this.mainAudioContext.destination);
            this.generatedSoundOscillator.type = 'sine';
            let freqToPlay = (this.drivingMechanism === 'piano' && (this.activePianoKeys.size > 0 || this.keyboardPressedKeys.size > 0)) ? this.currentFrequency : this.actualAppliedFrequency;
            if (freqToPlay > 0 && freqToPlay < 22000) {
                this.generatedSoundOscillator.frequency.setValueAtTime(freqToPlay, this.mainAudioContext.currentTime);
                this.generatedSoundGainNode.gain.setValueAtTime(0.15, this.mainAudioContext.currentTime);
                try { this.generatedSoundOscillator.start(); } catch (e) { this.isGeneratedSoundEnabled = false; }
            } else this.isGeneratedSoundEnabled = false;
        }
        this._updateUIToggleButtons();
        if (this.isGeneratedSoundEnabled) this._updateFrequencyControlsUI();
    }
    
    _getSpectralFlux(band) {
        if (!band.analyser || !this.mainAudioContext || this.mainAudioContext.state !== 'running') return 0;
        
        const bufferLength = band.analyser.frequencyBinCount;
        if (!band.frequencyData || band.frequencyData.length !== bufferLength) {
            band.frequencyData = new Uint8Array(bufferLength);
        }
        band.analyser.getByteFrequencyData(band.frequencyData);

        let flux = 0;
        for (let i = 0; i < bufferLength; i++) {
            const diff = band.frequencyData[i] - (band.previousData ? band.previousData[i] : 0);
            flux += diff > 0 ? diff : 0;
        }
        
        if (!band.previousData || band.previousData.length !== bufferLength) {
            band.previousData = new Uint8Array(bufferLength);
        }
        band.previousData.set(band.frequencyData);
        return flux;
    }
    
    _isLocalMaximum(fluxValue, history) {
        if (!history || history.length < 3) return false;
        const checkWindow = Math.min(Math.floor(history.length / 2), Math.floor(this.BPM_PEAK_DETECTION_WINDOW / 2));
        if (checkWindow < 1) return false;

        let isMax = true;
        for (let i = 1; i <= checkWindow; i++) {
            const pastIndex = history.length - 1 - i;
            if (pastIndex < 0) break;
            if (fluxValue <= history[pastIndex]) {
                isMax = false;
                break;
            }
        }
        return isMax;
    }

    _updateBPMThreshold(band) {
        const fluxHistoryForThreshold = band.history.slice(-50);
        if (fluxHistoryForThreshold && fluxHistoryForThreshold.length > 10) {
            const sortedFlux = [...fluxHistoryForThreshold].sort((a, b) => a - b);
            const percentileIndex = Math.floor(sortedFlux.length * 0.70);
            let newThreshold = sortedFlux[percentileIndex] * 0.9;
            newThreshold = Math.max(30, Math.min(newThreshold, 800));
            band.threshold = (band.threshold || 100) * 0.95 + newThreshold * 0.05;
        } else if (band.threshold) {
            band.threshold = Math.max(30, band.threshold * 0.99);
        }
    }
    
    _updateAndAnalyzeBPM() {
        if (!this.mainAudioContext || this.mainAudioContext.state !== 'running') return;
        
        const currentTime = this.mainAudioContext.currentTime;
        let newBeats = [];

        for (const bandName of ['low', 'mid', 'high']) {
            const band = this.bpmAnalyzers[bandName];
            if (!band.analyser) continue;

            const flux = this._getSpectralFlux(band);
            band.history.push(flux);
            if (band.history.length > 50) band.history.shift();

            this._updateBPMThreshold(band);
            const isPeak = this._isLocalMaximum(flux, band.history);

            if (isPeak && flux > band.threshold) {
                const lastBeatTime = this.allDetectedBeats.length > 0 ? this.allDetectedBeats[this.allDetectedBeats.length - 1].time : -1;
                if (currentTime - lastBeatTime < 0.22) {
                    continue;
                }
                newBeats.push({ time: currentTime, band: bandName, strength: flux });
            }
        }
        
        if (newBeats.length > 0) {
            this.allDetectedBeats.push(...newBeats);
            if (this.allDetectedBeats.length > 60) {
                this.allDetectedBeats = this.allDetectedBeats.slice(this.allDetectedBeats.length - 60);
            }
        }

        this._calculateCombinedBPM();
    }

    _calculateCombinedBPM() {
        if (this.allDetectedBeats.length < 8) {
            if (this.uiElements.bpmConfidence) this.uiElements.bpmConfidence.textContent = 'ожидание...';
            return;
        }

        const intervals = [];
        for (let i = 1; i < this.allDetectedBeats.length; i++) {
            const interval = this.allDetectedBeats[i].time - this.allDetectedBeats[i - 1].time;
            if (interval > 0.20 && interval < 2.0) {
                intervals.push({
                    interval: interval,
                    band: this.allDetectedBeats[i].band,
                    bpm: 60.0 / interval
                });
            }
        }

        if (intervals.length < 5) return;

        const bpmClusters = {};
        const bpmTolerance = 5;
        intervals.forEach(item => {
            const bpm = item.bpm;
            let foundCluster = false;
            for (const key in bpmClusters) {
                if (Math.abs(bpm - key) < bpmTolerance) {
                    bpmClusters[key].push(item);
                    foundCluster = true;
                    break;
                }
            }
            if (!foundCluster) {
                bpmClusters[bpm] = [item];
            }
        });

        let bestCandidate = null;
        let maxScore = -1;

        for (const key in bpmClusters) {
            const cluster = bpmClusters[key];
            if (cluster.length < 3) continue;

            let score = cluster.length;
            
            cluster.forEach(item => {
                if (item.band === 'low') score += BPM_LOW_BAND_WEIGHT;
                if (item.band === 'mid') score += BPM_MID_BAND_WEIGHT;
                if (item.band === 'high') score += BPM_HIGH_BAND_WEIGHT;
            });
            
            const clusterIntervals = cluster.map(item => item.interval).sort((a, b) => a - b);
            const meanInterval = clusterIntervals.reduce((sum, i) => sum + i, 0) / cluster.length;
            if (meanInterval > 0) {
                const intervalDeviations = clusterIntervals.map(item => Math.abs(item - meanInterval));
                const tempoConsistency = 1 - (intervalDeviations.reduce((sum, dev) => sum + dev, 0) / cluster.length) / meanInterval;
                score *= Math.max(0.1, tempoConsistency);
            }

            const doubleBPM = parseFloat(key) * 2;
            for (const otherKey in bpmClusters) {
                if (Math.abs(doubleBPM - otherKey) < bpmTolerance * 2) {
                    if (bpmClusters[otherKey].length > cluster.length / 2) {
                        score *= 0.7;
                    }
                }
            }

            if (score > maxScore) {
                maxScore = score;
                bestCandidate = cluster;
            }
        }

        if (bestCandidate) {
            const candidateIntervals = bestCandidate.map(item => item.interval).sort((a, b) => a - b);
            const medianInterval = candidateIntervals[Math.floor(candidateIntervals.length / 2)];
            
            if (medianInterval > 0) {
                const newBPM = 60.0 / medianInterval;
                const mad = candidateIntervals.reduce((sum, val) => sum + Math.abs(val - medianInterval), 0) / candidateIntervals.length;
                const confidence = Math.max(0, 1 - (mad / medianInterval));

                const alpha = Math.min(0.3, Math.abs(newBPM - this.currentBPM) / 50 + 0.05 + (1 - confidence) * 0.1);
                this.currentBPM = this.currentBPM * (1 - alpha) + newBPM * alpha;
                this.currentBPM = THREE.MathUtils.clamp(this.currentBPM, 40, 220);

                if (this.uiElements.bpmValue) this.uiElements.bpmValue.textContent = this.currentBPM.toFixed(1);
                if (this.uiElements.bpmConfidence) this.uiElements.bpmConfidence.textContent = `${(confidence * 100).toFixed(1)}%`;
            }
        }
    }

    _startBPMAnalysisLoop() {
        if (this.bpmUpdateIntervalId) clearInterval(this.bpmUpdateIntervalId);
        
        const sourceActive = (this.isAudioFilePlaying && !this.isAudioFilePaused) || this.isMicrophoneEnabled || this.isDesktopAudioEnabled;
        if (!sourceActive || !this.mainAudioContext || this.mainAudioContext.state !== 'running') return;
        
        this.allDetectedBeats = [];
        for (const bandName in this.bpmAnalyzers) {
            const band = this.bpmAnalyzers[bandName];
            band.history = [];
            if (band.previousData) band.previousData.fill(0);
            band.threshold = 100;
        }

        this.bpmUpdateIntervalId = setInterval(() => {
            const currentSourceActive = (this.isAudioFilePlaying && !this.isAudioFilePaused) || this.isMicrophoneEnabled || this.isDesktopAudioEnabled;
            if (currentSourceActive && this.mainAudioContext.state === 'running') {
                this._updateAndAnalyzeBPM();
            } else {
                this._stopBPMAnalysisLoop();
            }
        }, 80);
    }

    _stopBPMAnalysisLoop() {
        if (this.bpmUpdateIntervalId) {
            clearInterval(this.bpmUpdateIntervalId);
            this.bpmUpdateIntervalId = null;
        }
        if (this.bandSources) {
            try {
                this.bandSources.low.stop();
                this.bandSources.mid.stop();
                this.bandSources.high.stop();
            } catch (e) {}
            this.bandSources = null;
        }
    }
    
    async _handleAudioFileSelection(event) {
        if (this.isLoadingTrack) return;
        const files = event.target.files;
        if (!files || files.length === 0 || !this.mainAudioContext) {
            if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = "Файлы не выбраны.";
            return;
        }
    
        await this._stopLoadedAudioFilePlayback(true);
        if (this.isMicrophoneEnabled) await this._toggleMicrophoneInput();
        if (this.isDesktopAudioEnabled) await this._toggleDesktopAudio();
    
        this.playlistFiles = Array.from(files);
        this.currentPlaylistIndex = -1;
    
        if (this.uiElements.audioInfoEl) {
             this.uiElements.audioInfoEl.textContent = `Загружено ${this.playlistFiles.length} трек(ов). Нажмите "Воспр."`;
        }
        if (this.uiElements.playUploadedAudioButton) {
            this.uiElements.playUploadedAudioButton.style.display = 'block';
            this.uiElements.playUploadedAudioButton.disabled = false;
            this.uiElements.playUploadedAudioButton.textContent = this.playlistFiles.length > 1 ? 'Воспр. плейлист' : 'Воспр. трек';
        }
    }
    
    async _createFrequencyBands(originalBuffer) {
        if (!originalBuffer) return null;
    
        const offlineCtxOptions = {
            length: originalBuffer.length,
            sampleRate: originalBuffer.sampleRate,
            numberOfChannels: originalBuffer.numberOfChannels
        };
    
        const createBand = async (filterType, frequency, q) => {
            const ctx = new OfflineAudioContext(offlineCtxOptions);
            const source = ctx.createBufferSource();
            source.buffer = originalBuffer;
            const filter = ctx.createBiquadFilter();
            filter.type = filterType;
            filter.frequency.value = frequency;
            if (q) filter.Q.value = q;
    
            source.connect(filter);
            filter.connect(ctx.destination);
            source.start(0);
            return await ctx.startRendering();
        }

        try {
            const [low, mid, high] = await Promise.all([
                createBand('lowpass', 250, 0.7071),
                createBand('bandpass', 1500, 0.8),
                createBand('highpass', 4000, 0.7071)
            ]);
            return { low, mid, high };
        } catch (e) {
            return null;
        }
    }
    
    async _loadAndPlayTrack(trackIndex) {
        if (this.isLoadingTrack) return;
        if (trackIndex < 0 || trackIndex >= this.playlistFiles.length) {
            await this._stopLoadedAudioFilePlayback(true);
            return;
        }
        this.isLoadingTrack = true;
        this.currentPlaylistIndex = trackIndex;
        const file = this.playlistFiles[trackIndex];
        
        this.activeFetchID++;
        const currentFetchID = this.activeFetchID;
    
        if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = `Обработка: ${file.name}...`;
        if (this.uiElements.playUploadedAudioButton) this.uiElements.playUploadedAudioButton.disabled = true;
        
        this.currentSubtitles = [];
        if (this.uiElements.lyricsInfoEl) this.uiElements.lyricsInfoEl.textContent = '';
        const fallbackPlayer = document.getElementById('fallback-audio-player');
        if (fallbackPlayer) fallbackPlayer.remove();
    
        try {
            const arrayBuffer = await file.arrayBuffer();
            
            await this._findAndLoadLyrics(file, currentFetchID);

            try {
                const decodedBuffer = await this.mainAudioContext.decodeAudioData(arrayBuffer.slice(0));
                
                this.audioFileBuffer = decodedBuffer;
                this.audioFileDuration = decodedBuffer.duration;

                let bands = null;
                if (decodedBuffer.duration < MAX_AUDIO_DURATION_FOR_MULTIBAND_S) {
                    bands = await this._createFrequencyBands(decodedBuffer);
                } else {
                    if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'none';
                }
    
                if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = `Трек ${trackIndex + 1}/${this.playlistFiles.length}: ${file.name}`;
    
                this.audioPlaybackOffset = 0;
                this.isAudioFilePaused = false;
    
                if (this.bandSources) {
                    try { this.bandSources.low.stop(); this.bandSources.mid.stop(); this.bandSources.high.stop(); } catch (e) {}
                }

                if (bands) {
                    const sourceLow = this.mainAudioContext.createBufferSource(); sourceLow.buffer = bands.low; sourceLow.connect(this.bpmAnalyzers.low.analyser);
                    const sourceMid = this.mainAudioContext.createBufferSource(); sourceMid.buffer = bands.mid; sourceMid.connect(this.bpmAnalyzers.mid.analyser);
                    const sourceHigh = this.mainAudioContext.createBufferSource(); sourceHigh.buffer = bands.high; sourceHigh.connect(this.bpmAnalyzers.high.analyser);
                    this.bandSources = { low: sourceLow, mid: sourceMid, high: sourceHigh };
                } else {
                    this.bandSources = null;
                }
    
                if (this.audioElement) { this.audioElement.pause(); if (this.audioElement.src) URL.revokeObjectURL(this.audioElement.src); this.audioElement.remove(); }
                this.audioElement = document.createElement('audio');
                this.audioElement.src = URL.createObjectURL(file);
                
                if (this.audioFileSourceNode) { try { this.audioFileSourceNode.disconnect(); } catch (e) {} }
                this.audioFileSourceNode = this.mainAudioContext.createMediaElementSource(this.audioElement);
        
                this.audioFileSourceNode.connect(this.mainAudioContext.destination);
                if (this.pitchDetectorAnalyserNode) this.audioFileSourceNode.connect(this.pitchDetectorAnalyserNode);
                if (this.fftAnalyserNode) this.audioFileSourceNode.connect(this.fftAnalyserNode);
                
                this._setActiveDrivingMechanism('audio', true);
                await this._playCurrentAudioElement();

            } catch (decodeError) {
                if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = `Не удалось декодировать, используется базовый плеер.`;
                
                this._stopBPMAnalysisLoop();
                this.bandSources = null;
                if (this.audioElement) { this.audioElement.pause(); if (this.audioElement.src) URL.revokeObjectURL(this.audioElement.src); this.audioElement.remove(); }
                
                const fallbackAudio = document.createElement('audio');
                fallbackAudio.src = URL.createObjectURL(file);
                fallbackAudio.controls = true;
                fallbackAudio.id = 'fallback-audio-player';
                this.uiElements.audioInfoEl.parentNode.insertBefore(fallbackAudio, this.uiElements.lyricsInfoEl.nextSibling); 
                
                this.audioElement = fallbackAudio;
                this.audioFileDuration = 0;
                
                if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'none';
                if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'none';
                
                await fallbackAudio.play();
                this.isAudioFilePlaying = true;
                this.isAudioFilePaused = false;
                
                fallbackAudio.onloadedmetadata = () => { this.audioFileDuration = fallbackAudio.duration; };
                fallbackAudio.onended = () => { if (this.isAudioFilePlaying) this._nextTrack(); };

                if (this.uiElements.playUploadedAudioButton) this.uiElements.playUploadedAudioButton.style.display = 'none';
                if (this.uiElements.stopAudioButton) this.uiElements.stopAudioButton.style.display = 'block';
                if (this.uiElements.toggleAudioPauseButton) this.uiElements.toggleAudioPauseButton.style.display = 'none';
            }
        } catch (e) {
            if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = `Ошибка: ${file.name} не удалось загрузить.`;
            await this._nextTrack(true);
        } finally {
            this.isLoadingTrack = false;
            if (this.uiElements.playUploadedAudioButton) this.uiElements.playUploadedAudioButton.disabled = false;
        }
    }
    
    async _nextTrack(isAutoAdvanceOnError = false) {
        if (this.isLoadingTrack) return;
        if (this.currentPlaylistIndex + 1 < this.playlistFiles.length) {
            await this._loadAndPlayTrack(this.currentPlaylistIndex + 1);
        } else {
            await this._stopLoadedAudioFilePlayback(true);
            if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = "Плейлист завершен.";
        }
    }
    
    async _prevTrack() {
        if (this.isLoadingTrack) return;
        if (this.currentPlaylistIndex - 1 >= 0) {
            await this._loadAndPlayTrack(this.currentPlaylistIndex - 1);
        }
    }
        
    async _playCurrentAudioElement() {
        if (!this.audioElement || this.audioElement.controls) return;
        if (!this.mainAudioContext) return;
        
        if (this.mainAudioContext.state === 'suspended') {
            try { await this.mainAudioContext.resume(); }
            catch (e) { return; }
        }

        this.audioElement.currentTime = this.audioPlaybackOffset;
        try {
            if (this.bandSources) {
                this.bandSources.low.start(0, this.audioPlaybackOffset);
                this.bandSources.mid.start(0, this.audioPlaybackOffset);
                this.bandSources.high.start(0, this.audioPlaybackOffset);
            }

            await this.audioElement.play();
            this.isAudioFilePlaying = true;
            this.isAudioFilePaused = false;

            if (this.bandSources) {
                this._startBPMAnalysisLoop();
            }

            if (this.uiElements.playUploadedAudioButton) this.uiElements.playUploadedAudioButton.style.display = 'none';
            if (this.uiElements.stopAudioButton) this.uiElements.stopAudioButton.style.display = 'block';
            if (this.uiElements.toggleAudioPauseButton) { this.uiElements.toggleAudioPauseButton.style.display = 'block'; this.uiElements.toggleAudioPauseButton.textContent = 'Пауза'; }
            if (this.uiElements.nextTrackButton) this.uiElements.nextTrackButton.style.display = this.playlistFiles.length > 1 ? 'block' : 'none';
            if (this.uiElements.prevTrackButton) this.uiElements.prevTrackButton.style.display = this.playlistFiles.length > 1 ? 'block' : 'none';
            if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'block';
            if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = this.bandSources ? 'block' : 'none';
            if (this.uiElements.audioProgressSlider) this.uiElements.audioProgressSlider.style.display = 'block';
            if (this.uiElements.audioTimeDisplay) this.uiElements.audioTimeDisplay.style.display = 'inline';

            this.audioElement.onended = () => { if (this.isAudioFilePlaying) this._nextTrack(); };
            this.audioElement.onerror = (e) => { this._nextTrack(true); };

        } catch (e) {
            await this._nextTrack(true);
        }
    }

    async _stopLoadedAudioFilePlayback(isFullReset = false) {
        this._stopBPMAnalysisLoop();

        const fallbackPlayer = document.getElementById('fallback-audio-player');
        if (fallbackPlayer) {
            fallbackPlayer.pause();
            fallbackPlayer.remove();
        }

        if (this.audioFileSourceNode) {
            try { this.audioFileSourceNode.disconnect(); } catch (e) { }
            this.audioFileSourceNode = null;
        }
        if (this.audioElement) {
            this.audioElement.pause();
            this.audioElement.onended = null;
            this.audioElement.onerror = null;
            if (this.audioElement.src) URL.revokeObjectURL(this.audioElement.src);
            if (this.audioElement.parentElement) this.audioElement.remove();
            this.audioElement = null;
        }
        
        this.isAudioFilePlaying = false;
        this.isAudioFilePaused = false;
        this.audioFileCurrentTime = 0;
        this.audioPlaybackOffset = 0;
        this.currentSubtitles = [];
        this.subtitleContainer.classList.remove('visible');
        this.subtitleContainer.textContent = '';


        if (isFullReset) {
            this.playlistFiles = [];
            this.currentPlaylistIndex = -1;
            this.audioFileBuffer = null;
            if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = "Аудио не загружено";
            if (this.uiElements.lyricsInfoEl) this.uiElements.lyricsInfoEl.textContent = "";
            if (this.uiElements.playUploadedAudioButton) this.uiElements.playUploadedAudioButton.style.display = 'none';
            if (this.uiElements.nextTrackButton) this.uiElements.nextTrackButton.style.display = 'none';
            if (this.uiElements.prevTrackButton) this.uiElements.prevTrackButton.style.display = 'none';
        } else if (this.playlistFiles.length > 0 && this.currentPlaylistIndex !== -1) {
             if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = `Остановлено. Готов трек ${this.currentPlaylistIndex + 1}/${this.playlistFiles.length}`;
             if (this.uiElements.playUploadedAudioButton) {
                this.uiElements.playUploadedAudioButton.style.display = 'block';
                this.uiElements.playUploadedAudioButton.disabled = false;
             }
        }
        
        if (this.uiElements.stopAudioButton) this.uiElements.stopAudioButton.style.display = 'none';
        if (this.uiElements.toggleAudioPauseButton) this.uiElements.toggleAudioPauseButton.style.display = 'none';
        
        if (isFullReset) {
            if (this.uiElements.audioProgressSlider) this.uiElements.audioProgressSlider.style.display = 'none';
            if (this.uiElements.audioTimeDisplay) this.uiElements.audioTimeDisplay.style.display = 'none';
            if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'none';
            if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'none';
        }
        
        this.lastStablePitchFrequency = 0;
        if (this.drivingMechanism === 'audio') this._setActiveDrivingMechanism('modal');
    }
    
    async _togglePauseLoadedAudioFilePlayback() {
        if (!this.audioElement && this.currentPlaylistIndex === -1) return;
        if (this.isLoadingTrack) return;
    
        if (this.audioElement.controls) return;

        if (this.isAudioFilePaused) {
            this.isAudioFilePaused = false;
            
            if (this.audioFileSourceNode && this.audioFileDuration > 0) {
                if (this.audioFileDuration < MAX_AUDIO_DURATION_FOR_MULTIBAND_S) {
                    const bands = await this._createFrequencyBands(this.audioFileBuffer);
                    if (!bands) { return; }
                    
                    const sourceLow = this.mainAudioContext.createBufferSource(); sourceLow.buffer = bands.low; sourceLow.connect(this.bpmAnalyzers.low.analyser);
                    const sourceMid = this.mainAudioContext.createBufferSource(); sourceMid.buffer = bands.mid; sourceMid.connect(this.bpmAnalyzers.mid.analyser);
                    const sourceHigh = this.mainAudioContext.createBufferSource(); sourceHigh.buffer = bands.high; sourceHigh.connect(this.bpmAnalyzers.high.analyser);
                    this.bandSources = { low: sourceLow, mid: sourceMid, high: sourceHigh };
                }
                await this._playCurrentAudioElement();
            } else {
                await this.audioElement.play();
                this.isAudioFilePlaying = true;
            }

            if (this.uiElements.toggleAudioPauseButton) this.uiElements.toggleAudioPauseButton.textContent = 'Пауза';

        } else {
            if (!this.isAudioFilePlaying || !this.audioElement) return;
            this.isAudioFilePaused = true;
            this.isAudioFilePlaying = false;
            this.audioElement.pause();
            this.audioPlaybackOffset = this.audioElement.currentTime;
            this._stopBPMAnalysisLoop();
            if (this.uiElements.toggleAudioPauseButton) this.uiElements.toggleAudioPauseButton.textContent = 'Продолжить';
        }
    }

    _updateAudioFileProgressControlsUI() {
        if (!this.mainAudioContext || !this.audioElement) return;

        if ((this.isAudioFilePlaying && !this.isAudioFilePaused) || this.audioElement.controls) {
            this.audioFileCurrentTime = this.audioElement.currentTime;
        } else {
            this.audioFileCurrentTime = this.audioPlaybackOffset;
        }
        
        const duration = this.audioElement.duration || this.audioFileDuration || 0;
        if (!isFinite(duration) || duration === 0) return;

        this.audioFileCurrentTime = Math.max(0, Math.min(this.audioFileCurrentTime, duration));

        if (this.uiElements.audioProgressSlider) {
            if (document.activeElement !== this.uiElements.audioProgressSlider) {
                this.uiElements.audioProgressSlider.value = (this.audioFileCurrentTime / duration) * 100;
            }
            if (this.uiElements.audioTimeDisplay) {
                this.uiElements.audioTimeDisplay.textContent = `${this._formatTimeMMSS(this.audioFileCurrentTime)} / ${this._formatTimeMMSS(duration)}`;
            }
        }
    }

    async _toggleMicrophoneInput() {
        if (this.isLoadingTrack) return;
        if (!this.mainAudioContext) {
            if (this.uiElements.microphoneInfoEl) this.uiElements.microphoneInfoEl.textContent = "Аудиосистема не готова.";
            this._updateUIToggleButtons();
            return;
        }
        if (this.mainAudioContext.state === 'suspended') {
            try { await this.mainAudioContext.resume(); }
            catch (e) {
                if (this.uiElements.microphoneInfoEl) this.uiElements.microphoneInfoEl.textContent = "Ошибка аудиосистемы.";
                this._updateUIToggleButtons();
                return;
            }
        }

        if (this.isMicrophoneEnabled) {
            if (this.microphoneStream) {
                this.microphoneStream.getTracks().forEach(track => track.stop());
                this.microphoneStream = null;
            }
            if (this.microphoneSourceNode) {
                try { this.microphoneSourceNode.disconnect(); } catch (e) { }
                this.microphoneSourceNode = null;
            }
            this._stopBPMAnalysisLoop();
            if (this.uiElements.microphoneInfoEl) this.uiElements.microphoneInfoEl.textContent = "Микрофон выключен.";
            if (this.drivingMechanism === 'microphone') {
                 this._setActiveDrivingMechanism('modal');
            }
            if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'none';
            if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'none';
            this.isMicrophoneEnabled = false;

        } else {
            this.isMicrophoneEnabled = true;
            if (this.isAudioFilePlaying || this.isAudioFilePaused) await this._stopLoadedAudioFilePlayback(true);
            if (this.isGeneratedSoundEnabled) await this._toggleGeneratedSoundPlayback();
            if (this.isDesktopAudioEnabled) await this._toggleDesktopAudio();

            try {
                const constraints = { audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, latency: 0.01 } };
                this.microphoneStream = await navigator.mediaDevices.getUserMedia(constraints);
                this.microphoneSourceNode = this.mainAudioContext.createMediaStreamSource(this.microphoneStream);
                
                if (this.pitchDetectorAnalyserNode) this.microphoneSourceNode.connect(this.pitchDetectorAnalyserNode);
                
                const band = this.bpmAnalyzers.low;
                if (band.analyser) {
                    const lowpassFilter = this.mainAudioContext.createBiquadFilter();
                    lowpassFilter.type = "lowpass";
                    lowpassFilter.frequency.value = 250;
                    this.microphoneSourceNode.connect(lowpassFilter);
                    lowpassFilter.connect(band.analyser);
                }

                if (this.fftAnalyserNode) this.microphoneSourceNode.connect(this.fftAnalyserNode);
                
                this._setActiveDrivingMechanism('microphone', true);
                this._startBPMAnalysisLoop();
                if (this.uiElements.microphoneInfoEl) this.uiElements.microphoneInfoEl.textContent = "Микрофон активен.";
                if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'block';
                if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'block';
                this.lastStablePitchFrequency = 0;

            } catch (err) {
                console.error("Ошибка доступа к микрофону:", err);
                if (this.uiElements.microphoneInfoEl) this.uiElements.microphoneInfoEl.textContent = `Ошибка: ${err.name}. Проверьте разрешения.`;
                this.isMicrophoneEnabled = false;
                if (this.drivingMechanism === 'microphone') this._setActiveDrivingMechanism('modal');
            }
        }
        this._updateUIToggleButtons();
    }
    
    async _toggleDesktopAudio() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
            alert('Ваш браузер не поддерживает захват экрана/аудио.');
            return;
        }

        if (this.isDesktopAudioEnabled) {
            if (this.desktopStream) {
                this.desktopStream.getTracks().forEach(track => track.stop());
                this.desktopStream = null;
            }
            if (this.desktopAudioSourceNode) {
                try { this.desktopAudioSourceNode.disconnect(); } catch(e) {}
                this.desktopAudioSourceNode = null;
            }
            this._stopBPMAnalysisLoop();
            if (this.drivingMechanism === 'desktop_audio') {
                this._setActiveDrivingMechanism('modal');
            }
            if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'none';
            if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'none';
            
            this.isDesktopAudioEnabled = false;
            this._updateUIToggleButtons();
            return;
        }

        try {
            if (this.isAudioFilePlaying || this.isAudioFilePaused) await this._stopLoadedAudioFilePlayback(true);
            if (this.isGeneratedSoundEnabled) await this._toggleGeneratedSoundPlayback();
            if (this.isMicrophoneEnabled) await this._toggleMicrophoneInput();

            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: true,
                audio: true
            });
            
            const audioTrack = stream.getAudioTracks()[0];
            if (!audioTrack) {
                alert('Не удалось захватить аудио. Убедитесь, что вы поставили галочку "Поделиться системным аудио" в диалоговом окне выбора экрана.');
                stream.getTracks().forEach(track => track.stop());
                return;
            }
            
            this.isDesktopAudioEnabled = true;
            this.desktopStream = stream;
            this.desktopAudioSourceNode = this.mainAudioContext.createMediaStreamSource(this.desktopStream);

            if (this.pitchDetectorAnalyserNode) this.desktopAudioSourceNode.connect(this.pitchDetectorAnalyserNode);
            if (this.fftAnalyserNode) this.desktopAudioSourceNode.connect(this.fftAnalyserNode);
            
            const band = this.bpmAnalyzers.low;
            if (band.analyser) {
                const lowpassFilter = this.mainAudioContext.createBiquadFilter();
                lowpassFilter.type = "lowpass";
                lowpassFilter.frequency.value = 250;
                this.desktopAudioSourceNode.connect(lowpassFilter);
                lowpassFilter.connect(band.analyser);
            }

            this._setActiveDrivingMechanism('desktop_audio', true);
            this._startBPMAnalysisLoop();

            if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'block';
            if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'block';
            this.lastStablePitchFrequency = 0;
            
            audioTrack.onended = () => {
                if (this.isDesktopAudioEnabled) {
                    this._toggleDesktopAudio(); 
                }
            };
            
        } catch (err) {
            console.error("Ошибка захвата экрана/аудио:", err);
            this.isDesktopAudioEnabled = false;
            if (this.drivingMechanism === 'desktop_audio') this._setActiveDrivingMechanism('modal');
        } finally {
            this._updateUIToggleButtons();
        }
    }
    
    _handleKeyboardPiano(event){ 
        if (!this.uiElements.pianoContainer || 
            document.activeElement === this.uiElements.frequencyInput || 
            document.activeElement?.closest && (
                document.activeElement.closest('.control-panel input[type="number"]') || 
                document.activeElement.closest('.control-panel select') ||
                document.activeElement.closest('.control-panel input[type="file"]')
            )
        ) { return; } 
        const note = this.keyToNoteMapping[event.code]; 
        if(note){ 
            event.preventDefault(); 
            const isShiftPressed = event.shiftKey; 
            if(event.type === 'keydown' && !this.keyboardPressedKeys.has(event.code)){ 
                this._handlePianoKeyPress(event.code, true, true, isShiftPressed); 
            } else if (event.type === 'keyup' && this.keyboardPressedKeys.has(event.code)){ 
                this._handlePianoKeyPress(event.code, false, true, false); 
            } 
        } 
    }

    _createPianoKeys() { 
        if (!this.uiElements.pianoContainer) return; 
        this.uiElements.pianoContainer.innerHTML = ''; 
        const whiteW=35, blackW=22; 
        const keys = [{n:'C',t:'white'},{n:'C#',t:'black'},{n:'D',t:'white'},{n:'D#',t:'black'},{n:'E',t:'white'},{n:'F',t:'white'},{n:'F#',t:'black'},{n:'G',t:'white'},{n:'G#',t:'black'},{n:'A',t:'white'},{n:'A#',t:'black'},{n:'B',t:'white'}]; 
        let whiteOff=0; 
        keys.forEach(k=>{ 
            const div = document.createElement('div'); 
            div.className=`piano-key ${k.t}`; 
            div.textContent=k.n; 
            div.dataset.note=k.n; 
            if(k.t==='white'){
                div.style.left=`${whiteOff*whiteW}px`;
                whiteOff++;
            } else {
                div.style.left=`${(whiteOff-1)*whiteW+(whiteW-blackW/2)-1}px`;
            } 
            const press=(e)=>{e.preventDefault();this._handlePianoKeyPress(k.n,true, false, e.shiftKey);}; 
            const rel=(e)=>{e.preventDefault();this._handlePianoKeyPress(k.n,false, false, false);}; 
            div.addEventListener('mousedown',press); 
            div.addEventListener('mouseup',rel); 
            div.addEventListener('mouseleave',(e)=>{if(e.buttons===1&&this.activePianoKeys.has(k.n))this._handlePianoKeyPress(k.n,false, false, false);}); 
            div.addEventListener('touchstart',press,{passive:false}); 
            div.addEventListener('touchend',rel);
            div.addEventListener('touchcancel',rel); 
            this.uiElements.pianoContainer.appendChild(div); 
        }); 
    }

    async _handlePianoKeyPress(noteOrKeyCode, isPressed, isKeyboardEvent = false, isShiftPressed = false) {
        if (this.isLoadingTrack) return;
        let noteName = noteOrKeyCode;
        let actualOctave = this.currentPianoOctave;

        if (isKeyboardEvent) {
            if (isPressed && this.keyboardPressedKeys.has(noteOrKeyCode)) return;
            if (!isPressed && !this.keyboardPressedKeys.has(noteOrKeyCode)) return;
            noteName = this.keyToNoteMapping[noteOrKeyCode];
            if (!noteName) return;
            if (isPressed) this.keyboardPressedKeys.add(noteOrKeyCode);
            else this.keyboardPressedKeys.delete(noteOrKeyCode);

            if (isShiftPressed) actualOctave += 1; 

            if (noteName.endsWith('5')) {
                let baseOctave = this.currentPianoOctave + (noteName === 'C5' ? 1 : 0);
                actualOctave = baseOctave + (isShiftPressed ? 1:0); 
                noteName = noteName.slice(0, -1);
            } else if (noteName.length === 2 && noteName[1] === '5' && NOTE_TO_MIDI_NUMBER_OFFSET.hasOwnProperty(noteName[0])) {
                let baseOctave = this.currentPianoOctave + 1;
                actualOctave = baseOctave + (isShiftPressed ? 1:0);
                noteName = noteName[0];
            }
        } else { 
            if (isPressed) this.activePianoKeys.add(noteName);
            else this.activePianoKeys.delete(noteName);
            if (isShiftPressed && isPressed) actualOctave += 1;
        }
        
        actualOctave = Math.max(0, Math.min(8, actualOctave)); 

        const pianoKeyElement = this.uiElements.pianoContainer?.querySelector(`.piano-key[data-note="${noteName}"]`);
        if (pianoKeyElement) {
            let uiNoteName = noteName;
            if (noteOrKeyCode === 'KeyK' && noteName === 'C') uiNoteName = 'C';
            const baseNoteKeyElement = this.uiElements.pianoContainer?.querySelector(`.piano-key[data-note="${uiNoteName}"]`);
            if (baseNoteKeyElement) {
                if (isPressed) baseNoteKeyElement.classList.add('active');
                else {
                    const stillHeldByMouse = this.activePianoKeys.has(uiNoteName);
                    const stillHeldByKeyboard = Array.from(this.keyboardPressedKeys).some(kc => this.keyToNoteMapping[kc] === uiNoteName || (this.keyToNoteMapping[kc] && this.keyToNoteMapping[kc].startsWith(uiNoteName) && this.keyToNoteMapping[kc].endsWith('5')));
                    if (!stillHeldByMouse && !stillHeldByKeyboard) baseNoteKeyElement.classList.remove('active');
                }
            }
        }

        const allActiveNotes = new Set([...this.activePianoKeys, ...Array.from(this.keyboardPressedKeys).map(kc => { let mapped = this.keyToNoteMapping[kc]; if (mapped && mapped.endsWith('5')) mapped = mapped.slice(0,-1); return mapped;}).filter(Boolean)]);
        
        if (isPressed) { 
            this.currentFrequency = this._frequencyFromMIDINoteNumber(NOTE_TO_MIDI_NUMBER_OFFSET[noteName] + (actualOctave * 12));
            this.lastStablePitchFrequency = 0; 
            
            if (this.drivingMechanism !== 'piano') {
                this._setActiveDrivingMechanism('piano', true);
            } else { 
                this.actualAppliedFrequency = this.currentFrequency; 
                this._resetFullSimulationState(true); 
            }

            if (this.uiElements.pianoStatus) this.uiElements.pianoStatus.textContent = `Нота: ${noteName}${actualOctave} (${this.currentFrequency.toFixed(1)} Гц) FDM: ${this.actualAppliedFrequency.toFixed(1)}Гц`;
            
            if (this.isGeneratedSoundEnabled) {
                if (!this.mainAudioContext) this._setupWebAudioSystem();
                if (this.mainAudioContext?.state === 'suspended') await this.mainAudioContext.resume();

                if (this.generatedSoundOscillator) { 
                    try { this.generatedSoundOscillator.stop(this.mainAudioContext.currentTime); this.generatedSoundOscillator.disconnect(); } catch (e) {}
                    this.generatedSoundOscillator = null;
                }
                if (this.generatedSoundGainNode) {
                    try {this.generatedSoundGainNode.disconnect(); } catch (e) {}
                    this.generatedSoundGainNode = null;
                }

                this.generatedSoundOscillator = this.mainAudioContext.createOscillator();
                this.generatedSoundGainNode = this.mainAudioContext.createGain();
                this.generatedSoundOscillator.connect(this.generatedSoundGainNode);
                this.generatedSoundGainNode.connect(this.mainAudioContext.destination);
                this.generatedSoundOscillator.type = 'sine';
                const freqToPlay = Math.min(this.currentFrequency, 22000);
                if (freqToPlay > 0) {
                    this.generatedSoundOscillator.frequency.setValueAtTime(freqToPlay, this.mainAudioContext.currentTime);
                    this.generatedSoundGainNode.gain.setValueAtTime(0.15, this.mainAudioContext.currentTime);
                    try { this.generatedSoundOscillator.start(this.mainAudioContext.currentTime); } catch(e) {}
                }
            }

        } else if (allActiveNotes.size === 0) { 
            if (this.uiElements.pianoStatus) this.uiElements.pianoStatus.textContent = "Пианино неактивно";
            if (this.generatedSoundOscillator && this.isGeneratedSoundEnabled && this.mainAudioContext) {
                this.generatedSoundGainNode?.gain.setTargetAtTime(0, this.mainAudioContext.currentTime, 0.02);
                try {
                    const oscToStop = this.generatedSoundOscillator; 
                    const gainToStop = this.generatedSoundGainNode;
                    if (oscToStop) oscToStop.stop(this.mainAudioContext.currentTime + 0.05);

                    setTimeout(() => { 
                        if (oscToStop && oscToStop === this.generatedSoundOscillator) { 
                             try {oscToStop.disconnect();} catch(e){}
                             this.generatedSoundOscillator = null;
                        }
                        if (gainToStop && gainToStop === this.generatedSoundGainNode) {
                             try {gainToStop.disconnect();} catch(e){}
                             this.generatedSoundGainNode = null;
                        }
                    }, 60); 
                } catch(e) {}
            }
        }
    }
    
    _mapUIElements() { 
        const ids = [
            'frequencySlider', 'freqValueText', 'frequencyInput', 'setFrequencyButton', 'particleSpeedSlider', 'speedValueText', 'presetSelect', 'mParamSlider', 'mParamValueText', 'mParamInput', 'setMParamButton', 'nParamSlider', 'nParamValueText', 'nParamInput', 'setNParamButton',
            'toggleDesktopAudioButton',
            'toggleSoundButton', 'toggleFreezeButton', 'toggleSubtitlesButton', 'toggleShadowsButton', 'toggleDynamicDensityButton', 'toggleStuckParticleCullingButton', 'toggleFDMOptButton',
            'resetSimulationButton', 'simulationProgress', 'activeModeDisplay', 'plateRotationSpeedSlider', 'plateRotationSpeedValue', 'stopRotationButton',
            'audioFileInput', 'playUploadedAudioButton', 'stopAudioButton', 'toggleAudioPauseButton', 'audioInfoEl', 'lyricsInfoEl',
            'nextTrackButton', 'prevTrackButton', 
            'toggleMicrophoneButton', 'microphoneInfoEl',
            'audioProgressSlider', 'audioTimeDisplay', 'pitchDetectorInfo', 'pitch', 'note', 'detune_amt', 'detune', 'pianoOctaveSelect', 'pianoContainer', 'pianoStatus',
            'advPlateThickness', 'advPlateThicknessSlider', 'advPlateThicknessValue', 'advPlateDensity', 'advPlateDensitySlider', 'advPlateDensityValue', 'advEModulus', 'advEModulusSlider', 'advEModulusValue', 'advPoissonRatio', 'advPoissonRatioSlider', 'advPoissonRatioValue',
            'advMinGridSize', 'advMinGridSizeSlider', 'advMinGridSizeValue', 'advMaxGridSize', 'advMaxGridSizeSlider', 'advMaxGridSizeValue', 'advBaseMaxFDMSteps', 'advBaseMaxFDMStepsSlider', 'advBaseMaxFDMStepsValue', 'advFDMStabilityFactor', 'advFDMStabilityFactorSlider', 'advFDMStabilityFactorValue', 'advFDMDampingFactor', 'advFDMDampingFactorSlider', 'advFDMDampingFactorValue',
            'advBpmPeakWindow', 'advBpmPeakWindowSlider', 'advBpmPeakWindowValue', 
            'advParticleCount', 'advParticleCountSlider', 'advParticleCountValue', 'advParticleForceBase', 'advParticleForceBaseSlider', 'advParticleForceBaseValue', 'advParticleDampingBase', 'advParticleDampingBaseSlider', 'advParticleDampingBaseValue', 'advParticleSize', 'advParticleSizeSlider', 'advParticleSizeValue', 'advEnableRepulsion', 'advRepulsionRadius', 'advRepulsionRadiusSlider', 'advRepulsionRadiusValue', 'advRepulsionStrength', 'advRepulsionStrengthSlider', 'advRepulsionStrengthValue', 'advParticleRestitution', 'advMaxParticleSpeed', 'advMaxRepulsionNeighbors',
            'advExcBaseAmp', 'advExcBaseAmpSlider', 'advExcBaseAmpValue', 'advExcLowFreqCutoff', 'advExcHighFreqCutoff', 'advExcMaxFactor', 'advExcMinFactor',
            'advVisDeformScale', 'advVisDeformScaleSlider', 'advVisDeformScaleValue', 'advMaxVisAmplitude', 'advMaxVisAmplitudeSlider', 'advMaxVisAmplitudeValue',
            'resetAdvancedButton', 'toggleLeftPanelButton', 'toggleRightPanelButton', 'controls', 'advanced-controls',
            'bpmInfo', 'bpmValue', 'bpmConfidence', 'subtitle-container',
            'tooltip', 'welcome-overlay', 'prompt-overlay', 'welcome-modal', 'prompt-modal', 
            'lang-toggle-btn', 'show-prompt-btn', 'close-modal-btn', 'prompt-textarea', 
            'close-prompt-btn', 'show-welcome-btn', 'welcome-body-content'
        ];
        ids.forEach(id => {
            if (id) this.uiElements[id] = document.getElementById(id);
        }); 
        if (!this.uiElements['advanced-controls'] && this.uiElements.advancedcontrols) this.uiElements['advanced-controls'] = this.uiElements.advancedcontrols; 
    }
    
    _updateUIToggleButtons() {
        const setButtonState = (button, flag, onKeySuffix = '_on', offKeySuffix = '_off') => {
            if (!button) return;
            const langKey = button.dataset.langKey;
            
            if (langKey.startsWith('btn_')) {
                const textKey = flag ? `${langKey}${onKeySuffix}` : `${langKey}${offKeySuffix}`;
                 if (LANG_PACK[this.currentLanguage] && LANG_PACK[this.currentLanguage][textKey]) {
                    button.innerHTML = LANG_PACK[this.currentLanguage][textKey];
                }
            }
            
            button.classList.toggle('button-on', flag);
            button.classList.toggle('button-off', !flag);
        };

        setButtonState(this.uiElements.toggleDesktopAudioButton, this.isDesktopAudioEnabled);
        setButtonState(this.uiElements.toggleSoundButton, this.isGeneratedSoundEnabled);
        setButtonState(this.uiElements.toggleFreezeButton, this.areParticlesFrozen);
        setButtonState(this.uiElements.toggleSubtitlesButton, this.isSubtitlesEnabled);
        setButtonState(this.uiElements.toggleFDMOptButton, this.enableFDMOptimization);
        setButtonState(this.uiElements.toggleShadowsButton, this.enableShadows);
        setButtonState(this.uiElements.toggleDynamicDensityButton, this.enableDynamicParticleDensity);
        setButtonState(this.uiElements.toggleMicrophoneButton, this.isMicrophoneEnabled);
        setButtonState(this.uiElements.toggleStuckParticleCullingButton, this.enableStuckParticleCulling);
    }

    _updateParticleColorBasedOnFrequency() {
        if (!this.particlesMesh?.material.uniforms.globalColor) return;
        const freq = this.actualAppliedFrequency || this.currentFrequency;
        if (!freq || freq <= 0 || !isFinite(freq)) return;
        const minLog = Math.log10(20);
        const maxLog = Math.log10(20000);
        const currLog = Math.log10(Math.max(20, freq));
        let hue = THREE.MathUtils.clamp((currLog - minLog) / (maxLog - minLog + 1e-9), 0, 1) * 0.7;
        const color = new THREE.Color();
        
        color.setHSL(hue, 0.98, 0.58);
        this.particlesMesh.material.uniforms.globalColor.value.copy(color);
    }

    _updateFrequencyControlsUI() {
        if (this.uiElements.freqValueText) this.uiElements.freqValueText.textContent = `${this.currentFrequency.toFixed(0)} Гц`;
        if (this.uiElements.frequencyInput && document.activeElement !== this.uiElements.frequencyInput) this.uiElements.frequencyInput.value = this.currentFrequency.toFixed(1);
        if (this.uiElements.frequencySlider && document.activeElement !== this.uiElements.frequencySlider) {
            const minLogS = 20; const maxLogS = 20000; let valOut = 0;
            if (this.currentFrequency > minLogS) valOut = 100 * (Math.log10(this.currentFrequency / minLogS) / Math.log10(maxLogS / minLogS));
            this.uiElements.frequencySlider.value = THREE.MathUtils.clamp(valOut, 0, 100);
        }
        if (!this.isAudioFilePlaying && !this.isMicrophoneEnabled && this.isGeneratedSoundEnabled && this.generatedSoundOscillator && this.mainAudioContext?.state === 'running') {
            let freqToPlay = (this.drivingMechanism === 'piano' && (this.activePianoKeys.size > 0 || this.keyboardPressedKeys.size > 0)) ? this.currentFrequency : this.actualAppliedFrequency;
            if (freqToPlay > 0 && freqToPlay < this.mainAudioContext.sampleRate / 2) this.generatedSoundOscillator.frequency.setTargetAtTime(freqToPlay, this.mainAudioContext.currentTime, 0.01);
        }
        this._updateParticleColorBasedOnFrequency();
    }
    
    _updateModalParametersUI() { 
        if (this.uiElements.mParamValueText) this.uiElements.mParamValueText.textContent = this.mParameter; 
        if (this.uiElements.mParamInput && document.activeElement !== this.uiElements.mParamInput) this.uiElements.mParamInput.value = this.mParameter; 
        if (this.uiElements.mParamSlider && document.activeElement !== this.uiElements.mParamSlider) this.uiElements.mParamSlider.value = this.mParameter; 
        if (this.uiElements.nParamValueText) this.uiElements.nParamValueText.textContent = this.nParameter; 
        if (this.uiElements.nParamInput && document.activeElement !== this.uiElements.nParamInput) this.uiElements.nParamInput.value = this.nParameter; 
        if (this.uiElements.nParamSlider && document.activeElement !== this.uiElements.nParamSlider) this.uiElements.nParamSlider.value = this.nParameter; 
        if (this.uiElements.presetSelect) {
            const opt = this.uiElements.presetSelect.querySelector('option[value="none"]'); 
            if (opt && LANG_PACK[this.currentLanguage]) {
                if (this.drivingMechanism === 'modal') {
                    const currentPresetVal = `${this.mParameter},${this.nParameter}`; 
                    const match = Array.from(this.uiElements.presetSelect.options).find(o => o.value === currentPresetVal); 
                    if (match) {
                        this.uiElements.presetSelect.value = currentPresetVal; 
                    } else {
                        this.uiElements.presetSelect.value = "none";
                        opt.textContent = LANG_PACK[this.currentLanguage]['preset_custom'].replace('{m}', this.mParameter).replace('{n}', this.nParameter);
                    }
                } else {
                    this.uiElements.presetSelect.value = "none";
                    opt.textContent = LANG_PACK[this.currentLanguage]['preset_custom_inactive'];
                } 
            }
        } 
    }
    
    async _setActiveDrivingMechanism(newMechanism, keepParticlePositions = false) {
        const oldMechanism = this.drivingMechanism;
        this.drivingMechanism = newMechanism;

        if (this.uiElements.activeModeDisplay) {
            const key = `mode_${newMechanism.toLowerCase().replace('point', 'point')}`;
            if (LANG_PACK[this.currentLanguage] && LANG_PACK[this.currentLanguage][key]) {
                 this.uiElements.activeModeDisplay.innerHTML = LANG_PACK[this.currentLanguage][key];
            }
        }
        
        this._setupPlateParametersForCurrentMode();
        this._initializeFDM_CPU_State();
        this._initializeFDMArraysAndBuffers();
        this.simulationTime = 0;
        if (!keepParticlePositions) this._resetParticlePositionsAndVelocities();
        this._updateModalParametersUI();

        if (newMechanism === 'audio' || newMechanism === 'microphone' || newMechanism === 'desktop_audio') {
            if (this.isGeneratedSoundEnabled && this.generatedSoundOscillator) {
                await this._toggleGeneratedSoundPlayback();
            }
        }

        if ((oldMechanism === 'audio' || oldMechanism === 'microphone' || oldMechanism === 'desktop_audio') && (newMechanism !== 'audio' && newMechanism !== 'microphone' && newMechanism !== 'desktop_audio')) {
            this._stopBPMAnalysisLoop();
            this.currentBPM = 120;
            if (this.EXCITATION_FREQ_DEP_BASE_AMP !== this.normalExcBaseAmp) this.EXCITATION_FREQ_DEP_BASE_AMP = this.normalExcBaseAmp;
            if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'none';
            if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'none';
            this.lastStablePitchFrequency = 0;
        }
        
        if (newMechanism === 'audio' || newMechanism === 'microphone' || newMechanism === 'desktop_audio') {
            const sourceActive = (newMechanism === 'audio' && this.isAudioFilePlaying && !this.isAudioFilePaused) || (newMechanism === 'microphone' && this.isMicrophoneEnabled) || (newMechanism === 'desktop_audio' && this.isDesktopAudioEnabled);
            if (sourceActive) this._startBPMAnalysisLoop();
            if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'block';
            if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'block';
        } else {
             if (this.uiElements.bpmInfo) this.uiElements.bpmInfo.style.display = 'none';
             if (this.uiElements.pitchDetectorInfo) this.uiElements.pitchDetectorInfo.style.display = 'none';
        }
    }
    
    _populateAdvancedControlsUI() {
        const setCtrl = (id, val, type, exp, fix) => {
            const iE = this.uiElements[id]; 
            const sE = this.uiElements[id + "Slider"]; 
            const vE = this.uiElements[id + "Value"]; 
            let dS = (val == null) ? 'N/A' : ''; 

            if (val != null) {
                if (type === 'float') {
                    if (exp != null && (Math.abs(val) > 1e5 || (Math.abs(val) < 1e-3 && val !== 0))) {
                        dS = val.toExponential(exp);
                    } else if (fix != null) {
                        dS = val.toFixed(fix);
                    } else {
                        dS = val.toString();
                    }
                } else if (type === 'int') {
                    dS = val.toString();
                }
            }

            if (iE) {
                if (iE.type === 'checkbox') {
                    iE.checked = !!val;
                } else { 
                    let formattedModelValue;
                    if (type === 'float' && fix != null && typeof val === 'number') {
                        formattedModelValue = val.toFixed(fix);
                    } else if (type === 'int' && typeof val === 'number') {
                        formattedModelValue = val.toString();
                    } else { 
                        formattedModelValue = (val != null) ? val.toString() : '';
                    }
                     if (document.activeElement !== iE) iE.value = formattedModelValue;
                }
            }
            if (sE && val != null && isFinite(val)) {
                 if (document.activeElement !== sE) {
                    const sliderMin = parseFloat(sE.min);
                    const sliderMax = parseFloat(sE.max);
                    if (val >= sliderMin && val <= sliderMax) {
                        sE.value = val;
                    } else { 
                        sE.value = (val < sliderMin) ? sliderMin : sliderMax;
                    }
                }
            } else if (sE && document.activeElement !== sE) {
                sE.value = sE.min || 0;
            }
            if (vE) vE.textContent = dS;
        };

        setCtrl('advPlateThickness', this.PLATE_THICKNESS, 'float', null, 4);
        setCtrl('advPlateDensity', this.PLATE_DENSITY, 'int');
        setCtrl('advEModulus', this.E_MODULUS, 'float', 2);
        setCtrl('advPoissonRatio', this.POISSON_RATIO, 'float', null, 2);
        setCtrl('advMinGridSize', this.MIN_GRID_SIZE, 'int');
        setCtrl('advMaxGridSize', this.MAX_GRID_SIZE, 'int');
        setCtrl('advBaseMaxFDMSteps', this.BASE_MAX_FDM_STEPS_PER_FRAME, 'int');
        setCtrl('advFDMStabilityFactor', this.FDM_STABILITY_FACTOR, 'float', null, 4);
        setCtrl('advFDMDampingFactor', this.FDM_DAMPING_FACTOR, 'float', null, 6);
        setCtrl('advBpmPeakWindow', this.BPM_PEAK_DETECTION_WINDOW, 'int'); 
        setCtrl('advParticleCount', this.MAX_PARTICLE_COUNT_USER_SETTING, 'int'); 
        setCtrl('advParticleForceBase', this.PARTICLE_EFFECTIVE_FORCE_MULTIPLIER_BASE, 'float', 1);
        setCtrl('advParticleDampingBase', this.PARTICLE_DAMPING_BASE, 'float', null, 3);
        if (this.uiElements.advParticleRestitution && document.activeElement !== this.uiElements.advParticleRestitution) this.uiElements.advParticleRestitution.value = this.PARTICLE_BOUNDARY_RESTITUTION.toFixed(2);
        if (this.uiElements.advMaxParticleSpeed && document.activeElement !== this.uiElements.advMaxParticleSpeed) this.uiElements.advMaxParticleSpeed.value = this.MAX_PARTICLE_SPEED.toFixed(0);
        setCtrl('advParticleSize', this.PARTICLE_SIZE, 'float', null, 3);
        if(this.uiElements.advEnableRepulsion) setCtrl('advEnableRepulsion', this.ENABLE_PARTICLE_REPULSION, 'checkbox');
        setCtrl('advRepulsionRadius', this.PARTICLE_REPULSION_RADIUS, 'float', null, 3);
        setCtrl('advRepulsionStrength', this.PARTICLE_REPULSION_STRENGTH, 'float', null, 4);
        if (this.uiElements.advMaxRepulsionNeighbors && document.activeElement !== this.uiElements.advMaxRepulsionNeighbors) this.uiElements.advMaxRepulsionNeighbors.value = this.MAX_REPULSION_NEIGHBORS_CHECK;
        setCtrl('advExcBaseAmp', this.EXCITATION_FREQ_DEP_BASE_AMP, 'float', 1);
        if (this.uiElements.advExcLowFreqCutoff && document.activeElement !== this.uiElements.advExcLowFreqCutoff) this.uiElements.advExcLowFreqCutoff.value = this.EXCITATION_FREQ_DEP_LOW_CUTOFF;
        if (this.uiElements.advExcHighFreqCutoff && document.activeElement !== this.uiElements.advExcHighFreqCutoff) this.uiElements.advExcHighFreqCutoff.value = this.EXCITATION_FREQ_DEP_HIGH_CUTOFF;
        if (this.uiElements.advExcMaxFactor && document.activeElement !== this.uiElements.advExcMaxFactor) this.uiElements.advExcMaxFactor.value = this.EXCITATION_FREQ_DEP_MAX_FACTOR.toFixed(1);
        if (this.uiElements.advExcMinFactor && document.activeElement !== this.uiElements.advExcMinFactor) this.uiElements.advExcMinFactor.value = this.EXCITATION_FREQ_DEP_MIN_FACTOR.toFixed(1);
        setCtrl('advVisDeformScale', this.VISUAL_DEFORMATION_SCALE, 'float', null, 1);
        setCtrl('advMaxVisAmplitude', this.MAX_VISUAL_AMPLITUDE, 'float', null, 2);
    }
    
    _applyAdvancedSettingChange(paramName, valStrBool) { 
        let newVal; 
        const inputElement = this.uiElements[paramName] || this.uiElements[paramName.replace(/([A-Z])/g, (match, p1) => `-${p1.toLowerCase()}`).replace("adv-","adv")] ; 
        const minAttr = inputElement ? inputElement.min : ''; 
        const maxAttr = inputElement ? inputElement.max : ''; 
        const stepAttr = inputElement ? inputElement.step : ''; 
        const min = (minAttr !== '' && !isNaN(parseFloat(minAttr))) ? parseFloat(minAttr) : -Infinity; 
        const max = (maxAttr !== '' && !isNaN(parseFloat(maxAttr))) ? parseFloat(maxAttr) : Infinity; 
        const step = (stepAttr !== '' && stepAttr !== 'any' && !isNaN(parseFloat(stepAttr))) ? parseFloat(stepAttr) : (paramName.includes("GridSize") ? 2 : (paramName === "BPM_PEAK_DETECTION_WINDOW" ? 1 : 0.00000001) ); 
        
        if (typeof valStrBool === 'boolean' && inputElement?.type === 'checkbox') { newVal = valStrBool; } 
        else { 
            const isIntParam = paramName.includes("GridSize") || 
                               paramName.includes("Steps") || 
                               paramName.endsWith("Count") || 
                               paramName.includes("Density") || 
                               paramName.includes("Neighbors") || 
                               paramName.endsWith("Cutoff") || 
                               paramName === 'MAX_PARTICLE_SPEED' || 
                               paramName === 'MAX_REPULSION_NEIGHBORS_CHECK' || 
                               paramName === 'BASE_MAX_FDM_STEPS_PER_FRAME' || 
                               paramName === 'PARTICLE_COUNT' || 
                               paramName === 'MAX_PARTICLE_COUNT_USER_SETTING' ||
                               paramName === 'BPM_PEAK_DETECTION_WINDOW'; 
            newVal = this._parseInputNumber(valStrBool, this[paramName] , isIntParam, min, max); 
            if (inputElement?.type === "range" && step > 0 && !isIntParam) { 
                 newVal = Math.round(newVal / step) * step; 
                 newVal = parseFloat(newVal.toFixed(step.toString().split('.')[1]?.length || 7)); 
                 newVal = Math.max(min, Math.min(max, newVal)); 
            } else if (isIntParam && step > 0) {
                 newVal = Math.round(newVal / step) * step;
                 newVal = Math.max(min, Math.min(max, newVal));
            }
        } 
        let partRecr = false, critReset = false; 
        const previousValue = this[paramName]; 
        let wouldChange = ( Math.abs(previousValue - newVal) > 1e-9 || (typeof previousValue !== typeof newVal) ); 
        if(typeof previousValue === 'boolean') wouldChange = (previousValue !== newVal);

        if (paramName === 'MAX_PARTICLE_COUNT_USER_SETTING') {
            this.MAX_PARTICLE_COUNT_USER_SETTING = newVal;
            if (!this.enableDynamicParticleDensity) { 
                if (this.PARTICLE_COUNT !== newVal) {
                    this.PARTICLE_COUNT = newVal;
                    partRecr = true; 
                    critReset = true;
                }
            } else { 
                this._updateDynamicParticleDensity(); 
            }
        } else {
            this[paramName] = newVal;
        }

        if (paramName.includes("GridSize")) { 
            this[paramName] = this._roundToOddInteger(this[paramName]); 
            if (paramName === 'MIN_GRID_SIZE' && this.MIN_GRID_SIZE >= this.MAX_GRID_SIZE) { 
                this.MAX_GRID_SIZE = this._roundToOddInteger(this.MIN_GRID_SIZE + 2); 
            } else if (paramName === 'MAX_GRID_SIZE' && this.MAX_GRID_SIZE <= this.MIN_GRID_SIZE) { 
                this.MIN_GRID_SIZE = this._roundToOddInteger(this.MAX_GRID_SIZE - 2); 
                if (this.MIN_GRID_SIZE < (parseFloat(this.uiElements?.advMinGridSizeSlider?.min) || 11) ) this.MIN_GRID_SIZE = (parseFloat(this.uiElements?.advMinGridSizeSlider?.min) || 11); 
            } 
        } 
        if(this[paramName] !== newVal && paramName !== 'MAX_PARTICLE_COUNT_USER_SETTING') wouldChange = true; 
        newVal = this[paramName]; 
        if (paramName === 'EXCITATION_FREQ_DEP_BASE_AMP') {
            this.normalExcBaseAmp = newVal;
        }
        
        switch(paramName){ 
            case 'PLATE_THICKNESS': case 'PLATE_DENSITY': case 'E_MODULUS': case 'POISSON_RATIO': 
            case 'MIN_GRID_SIZE': case 'MAX_GRID_SIZE': case 'FDM_STABILITY_FACTOR': 
                critReset = true; break; 
            case 'BASE_MAX_FDM_STEPS_PER_FRAME': case 'FDM_DAMPING_FACTOR': case 'BPM_PEAK_DETECTION_WINDOW': break;  
            case 'PARTICLE_SIZE': 
                partRecr = true; critReset = true; break;  
        } 
        if (wouldChange || partRecr || critReset) { 
            this._updatePhysicalConstants(); 
            if (partRecr) this._createParticleSystem(); 
            if (critReset) this._resetFullSimulationState(true);  
        } 
        this._populateAdvancedControlsUI();  
    }
    
    _setupAdvancedControlsListeners() { 
        if(this.uiElements.resetAdvancedButton)this.uiElements.resetAdvancedButton.addEventListener('click',()=>this._resetAllSettingsToDefaults(true));
        
        const allConfigs = [ 
            {id:'advPlateThickness',varName:'PLATE_THICKNESS',type:'float',fixed:4},{id:'advPlateDensity',varName:'PLATE_DENSITY',type:'int'},{id:'advEModulus',varName:'E_MODULUS',type:'float',exp:2},{id:'advPoissonRatio',varName:'POISSON_RATIO',type:'float',fixed:2},{id:'advMinGridSize',varName:'MIN_GRID_SIZE',type:'int'},{id:'advMaxGridSize',varName:'MAX_GRID_SIZE',type:'int'},{id:'advBaseMaxFDMSteps',varName:'BASE_MAX_FDM_STEPS_PER_FRAME',type:'int'},{id:'advFDMStabilityFactor',varName:'FDM_STABILITY_FACTOR',type:'float',fixed:4},{id:'advFDMDampingFactor',varName:'FDM_DAMPING_FACTOR',type:'float',fixed:6},
            {id:'advBpmPeakWindow', varName:'BPM_PEAK_DETECTION_WINDOW', type:'int'},
            {id:'advParticleCount',varName:'MAX_PARTICLE_COUNT_USER_SETTING',type:'int'}, 
            {id:'advParticleForceBase',varName:'PARTICLE_EFFECTIVE_FORCE_MULTIPLIER_BASE',type:'float',exp:1},{id:'advParticleDampingBase',varName:'PARTICLE_DAMPING_BASE',type:'float',fixed:3},{id:'advParticleRestitution',varName:'PARTICLE_BOUNDARY_RESTITUTION',type:'float',fixed:2,noSlider:true},{id:'advMaxParticleSpeed',varName:'MAX_PARTICLE_SPEED',type:'float',fixed:0,noSlider:true},{id:'advParticleSize',varName:'PARTICLE_SIZE',type:'float',fixed:3},{id:'advEnableRepulsion',varName:'ENABLE_PARTICLE_REPULSION',type:'checkbox',noSlider:true},{id:'advRepulsionRadius',varName:'PARTICLE_REPULSION_RADIUS',type:'float',fixed:3},{id:'advRepulsionStrength',varName:'PARTICLE_REPULSION_STRENGTH',type:'float',fixed:4},{id:'advMaxRepulsionNeighbors',varName:'MAX_REPULSION_NEIGHBORS_CHECK',type:'int',noSlider:true},{id:'advExcBaseAmp',varName:'EXCITATION_FREQ_DEP_BASE_AMP',type:'float',exp:1},{id:'advExcLowFreqCutoff',varName:'EXCITATION_FREQ_DEP_LOW_CUTOFF',type:'int',noSlider:true},{id:'advExcHighFreqCutoff',varName:'EXCITATION_FREQ_DEP_HIGH_CUTOFF',type:'int',noSlider:true},{id:'advExcMaxFactor',varName:'EXCITATION_FREQ_DEP_MAX_FACTOR',type:'float',fixed:1,noSlider:true},{id:'advExcMinFactor',varName:'EXCITATION_FREQ_DEP_MIN_FACTOR',type:'float',fixed:1,noSlider:true},{id:'advVisDeformScale',varName:'VISUAL_DEFORMATION_SCALE',type:'float',fixed:1},{id:'advMaxVisAmplitude',varName:'MAX_VISUAL_AMPLITUDE',type:'float',fixed:2}
        ];

        const this_class = this; 

        allConfigs.forEach(c => { 
            const iE = this.uiElements[c.id]; 
            const sE = c.noSlider ? null : this.uiElements[c.id + "Slider"]; 
            const vE = this.uiElements[c.id + "Value"];
            
            const handleChange = (valFromUI) => {
                this_class._applyAdvancedSettingChange(c.varName, valFromUI);
            };

            if (iE) {
                const evT = (iE.type === 'checkbox') ? 'change' : 'input'; 
                iE.addEventListener(evT, function(event) { 
                    let valueToPass = c.type === 'checkbox' ? this.checked : this.value;
                    handleChange(valueToPass); 
                    
                    if (sE && document.activeElement !== sE) { 
                        const modelValue = parseFloat(this_class[c.varName]);
                        if (!isNaN(modelValue)) {
                            const sliderMin = parseFloat(sE.min);
                            const sliderMax = parseFloat(sE.max);
                            sE.value = THREE.MathUtils.clamp(modelValue, sliderMin, sliderMax);
                        }
                    }
                    if (event.type === 'change' && iE.type === 'number') iE.blur(); 
                });
                 if (iE.type === 'number') { 
                    iE.addEventListener('change', function(event) { handleChange(this.value); });
                }
            }
            
            if (sE) {
                sE.addEventListener('input', function(event) { 
                    const sliderVal = parseFloat(this.value); 
                    
                    let displayString = "";
                    if (c.type === 'float') {
                        if (c.exp != null && (Math.abs(sliderVal) > 1e5 || (Math.abs(sliderVal) < 1e-3 && sliderVal !== 0))) {
                            displayString = sliderVal.toExponential(c.exp);
                        } else if (c.fixed != null) {
                            displayString = sliderVal.toFixed(c.fixed);
                        } else {
                            displayString = sliderVal.toString();
                        }
                    } else if (c.type === 'int') {
                        displayString = Math.round(sliderVal).toString();
                    } else { 
                        displayString = sliderVal.toString();
                    }

                    if (iE && iE.type !== 'checkbox' && document.activeElement !== iE) {
                         let valToSetToInput = sliderVal;
                        if (c.type === 'int') valToSetToInput = Math.round(sliderVal);
                        else if (c.fixed != null) valToSetToInput = sliderVal.toFixed(c.fixed);
                        else valToSetToInput = sliderVal.toString();
                        
                        iE.value = valToSetToInput;
                    }
                    if (vE) {
                        vE.textContent = displayString;
                    }
                }); 
                 sE.addEventListener('change', function(event){ 
                    handleChange(parseFloat(this.value));
                });
            } 
        });
    }

    _storeDefaultSimulationSettings() {
        this.defaultAdvancedSettings = {
            PLATE_THICKNESS: PLATE_THICKNESS_DEFAULT,
            PLATE_DENSITY: PLATE_DENSITY_DEFAULT,
            E_MODULUS: E_MODULUS_DEFAULT,
            POISSON_RATIO: POISSON_RATIO_DEFAULT,
            MIN_GRID_SIZE: MIN_GRID_SIZE_DEFAULT,
            MAX_GRID_SIZE: MAX_GRID_SIZE_DEFAULT,
            BASE_MAX_FDM_STEPS_PER_FRAME: BASE_FDM_STEPS_CPU_DEFAULT,
            FDM_STABILITY_FACTOR: FDM_STABILITY_FACTOR_DEFAULT,
            FDM_DAMPING_FACTOR: FDM_DAMPING_FACTOR_DEFAULT,
            BPM_PEAK_DETECTION_WINDOW: BPM_PEAK_DETECTION_WINDOW_DEFAULT,
            MAX_PARTICLE_COUNT_USER_SETTING: PARTICLE_COUNT_DEFAULT,
            PARTICLE_EFFECTIVE_FORCE_MULTIPLIER_BASE: PARTICLE_FORCE_BASE_DEFAULT,
            PARTICLE_DAMPING_BASE: PARTICLE_DAMPING_BASE_DEFAULT,
            PARTICLE_BOUNDARY_RESTITUTION: PARTICLE_RESTITUTION_DEFAULT,
            MAX_PARTICLE_SPEED: MAX_PARTICLE_SPEED_DEFAULT,
            PARTICLE_SIZE: PARTICLE_SIZE_DEFAULT,
            ENABLE_PARTICLE_REPULSION: ENABLE_REPULSION_DEFAULT,
            PARTICLE_REPULSION_RADIUS: REPULSION_RADIUS_DEFAULT,
            PARTICLE_REPULSION_STRENGTH: REPULSION_STRENGTH_DEFAULT,
            MAX_REPULSION_NEIGHBORS_CHECK: MAX_REPULSION_NEIGHBORS_DEFAULT,
            EXCITATION_FREQ_DEP_BASE_AMP: EXC_BASE_AMP_DEFAULT,
            EXCITATION_FREQ_DEP_LOW_CUTOFF: EXC_LOW_CUTOFF_DEFAULT,
            EXCITATION_FREQ_DEP_HIGH_CUTOFF: EXC_HIGH_CUTOFF_DEFAULT,
            EXCITATION_FREQ_DEP_MAX_FACTOR: EXC_MAX_FACTOR_DEFAULT,
            EXCITATION_FREQ_DEP_MIN_FACTOR: EXC_MIN_FACTOR_DEFAULT,
            VISUAL_DEFORMATION_SCALE: VISUAL_DEFORMATION_SCALE_DEFAULT,
            MAX_VISUAL_AMPLITUDE: MAX_VISUAL_AMPLITUDE_DEFAULT
        };
    }

    async _resetAllSettingsToDefaults(isAdvancedOnly = false) {
        if (!isAdvancedOnly) {
            this.enableFDMOptimization = ENABLE_FDM_OPTIMIZATION_DEFAULT;
            this.enableShadows = ENABLE_SHADOWS_DEFAULT;
            this.enableDynamicParticleDensity = ENABLE_DYNAMIC_PARTICLE_DENSITY_DEFAULT;
            this.enableStuckParticleCulling = ENABLE_STUCK_PARTICLE_CULLING_DEFAULT;
            this.isSubtitlesEnabled = ENABLE_SUBTITLES_DEFAULT;
        }

        const prevMPC = this.MAX_PARTICLE_COUNT_USER_SETTING;
        const prevPS = this.PARTICLE_SIZE;
        
        Object.keys(this.defaultAdvancedSettings).forEach(k => {
             if (k === 'MAX_PARTICLE_COUNT_USER_SETTING') {
                this.MAX_PARTICLE_COUNT_USER_SETTING = this.defaultAdvancedSettings[k];
            } else {
                this[k] = (k.includes("GridSize")) ? this._roundToOddInteger(this.defaultAdvancedSettings[k]) : this.defaultAdvancedSettings[k];
            }
        });
        
        this.normalExcBaseAmp = this.defaultAdvancedSettings.EXCITATION_FREQ_DEP_BASE_AMP;
        
        if (this.enableDynamicParticleDensity) {
            this.PARTICLE_COUNT = MIN_DYNAMIC_PARTICLE_COUNT;
        } else {
            this.PARTICLE_COUNT = this.MAX_PARTICLE_COUNT_USER_SETTING;
        }
        
        this._updatePhysicalConstants();
        if (prevMPC !== this.MAX_PARTICLE_COUNT_USER_SETTING || prevPS !== this.PARTICLE_SIZE || this.PARTICLE_COUNT !== (this.enableDynamicParticleDensity ? MIN_DYNAMIC_PARTICLE_COUNT : this.MAX_PARTICLE_COUNT_USER_SETTING)) {
            this._createParticleSystem();
        }
        
        if (!isAdvancedOnly) {
            this.mParameter = 0; this.nParameter = 1; this.currentFrequency = 273; this.actualAppliedFrequency = 273;
            this.smoothedPitchFrequency = 273;
            this.lastStablePitchFrequency = 0;

            this.particleSimulationSpeedScale = 1.0;
            if (this.uiElements.particleSpeedSlider) this.uiElements.particleSpeedSlider.value = 50;
            if (this.uiElements.speedValueText) this.uiElements.speedValueText.textContent = `${this.particleSimulationSpeedScale.toFixed(2)}x`;
            
            this.plateRotationSpeed = 0;
            if (this.uiElements.plateRotationSpeedSlider) this.uiElements.plateRotationSpeedSlider.value = 0;
            if (this.uiElements.plateRotationSpeedValue) this.uiElements.plateRotationSpeedValue.textContent = "0.00 об/сек";
            
            Object.keys(this.besselZerosCache).forEach(key => delete this.besselZerosCache[key]);
            
            if (this.isMicrophoneEnabled) await this._toggleMicrophoneInput();
            if (this.isDesktopAudioEnabled) await this._toggleDesktopAudio();
            await this._stopLoadedAudioFilePlayback(true);
            if (this.uiElements.audioFileInput) this.uiElements.audioFileInput.value = "";
            if (this.isGeneratedSoundEnabled) await this._toggleGeneratedSoundPlayback();

            this._setActiveDrivingMechanism('modal', false);
        }
        this._populateAdvancedControlsUI();
        this._updateUIToggleButtons();
        this._applyShadowSettings();
        this._resetFullSimulationState(false);
    }

    _resetFullSimulationState(keepParticlePositions = false) {
        this._setupPlateParametersForCurrentMode();
        this._initializeFDM_CPU_State();
        this._initializeFDMArraysAndBuffers();
        this.simulationTime = 0;
        if (!keepParticlePositions) {
            this._resetParticlePositionsAndVelocities();
        }
        this._updateParticleColorBasedOnFrequency();
    }

    _applyShadowSettings() {
        if (!this.renderer || !this.dirLight1 || !this.groundPlane) return;
        this.renderer.shadowMap.enabled = this.enableShadows;
        this.dirLight1.castShadow = this.enableShadows;
        if (this.particlesMesh) this.particlesMesh.castShadow = this.enableShadows;
        this.groundPlane.receiveShadow = this.enableShadows;
        this.groundPlane.visible = this.enableShadows;
        
        this.scene.traverse(child => {
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.needsUpdate = true);
                } else {
                    child.material.needsUpdate = true;
                }
            }
        });
    }
    
    _setupEventListeners() { 
        window.addEventListener('resize', this._handleWindowResize.bind(this), false);
        window.addEventListener('keydown', this._handleKeyboardPiano.bind(this)); 
        window.addEventListener('keyup', this._handleKeyboardPiano.bind(this)); 
        
        if(this.uiElements.frequencySlider) this.uiElements.frequencySlider.addEventListener('input', (e)=>{ const sliderValPercent = parseFloat(e.target.value); const minLogF = 20; const maxLogF = Math.max(20000, this.EXCITATION_FREQ_DEP_HIGH_CUTOFF * 2); this.currentFrequency = this._parseInputNumber(minLogF * Math.pow(maxLogF / minLogF, sliderValPercent / 100.0), 273, false, 1, 999999); this.lastStablePitchFrequency = 0; this._setActiveDrivingMechanism('point', true); }); 
        const freqInputHandler = () => { const inputValue = this._parseInputNumber(this.uiElements.frequencyInput.value, this.currentFrequency, false, 1, 999999); this.currentFrequency = inputValue; this.lastStablePitchFrequency = 0; this._setActiveDrivingMechanism('point', true); }; 
        if(this.uiElements.setFrequencyButton)this.uiElements.setFrequencyButton.addEventListener('click',freqInputHandler); 
        if(this.uiElements.frequencyInput)this.uiElements.frequencyInput.addEventListener('change',freqInputHandler); 
        
        if(this.uiElements.presetSelect) this.uiElements.presetSelect.addEventListener('change', (e)=>{ const val = e.target.value; if(val === 'zvezda_lada'){this.mParameter=4; this.nParameter=2;} else if(val !== "none"){const p=val.split(','); this.mParameter=this._parseInputNumber(p[0],0,true,0,99); this.nParameter=this._parseInputNumber(p[1],1,true,1,99);} this.lastStablePitchFrequency = 0; this._setActiveDrivingMechanism('modal', true); if(this.uiElements.presetSelect)this.uiElements.presetSelect.blur(); }); 
        const mParamHandler=(valStr)=>{let v=this._parseInputNumber(valStr,this.mParameter,true,0,99);const maxM=parseInt(this.uiElements.mParamInput?.max)||99;v=Math.min(v,maxM);if(this.mParameter!==v){this.mParameter=v;this.nParameter=Math.max(1,Math.min(this.nParameter,parseInt(this.uiElements.nParamInput?.max)||99)); this.lastStablePitchFrequency = 0; this._setActiveDrivingMechanism('modal',true);}else this._updateModalParametersUI();}; 
        if(this.uiElements.mParamSlider)this.uiElements.mParamSlider.addEventListener('input',(e)=>mParamHandler(e.target.value)); 
        if(this.uiElements.setMParamButton && this.uiElements.mParamInput)this.uiElements.setMParamButton.addEventListener('click',()=>{mParamHandler(this.uiElements.mParamInput.value);this.uiElements.mParamInput.blur();}); 
        if(this.uiElements.mParamInput)this.uiElements.mParamInput.addEventListener('change',(e)=>mParamHandler(e.target.value)); 
        const nParamHandler=(valStr)=>{let v=this._parseInputNumber(valStr,this.nParameter,true,1,99);const maxN=parseInt(this.uiElements.nParamInput?.max)||99;v=Math.min(v,maxN);if(this.nParameter!==v){this.nParameter=v; this.lastStablePitchFrequency = 0; this._setActiveDrivingMechanism('modal',true);}else this._updateModalParametersUI();}; 
        if(this.uiElements.nParamSlider)this.uiElements.nParamSlider.addEventListener('input',(e)=>nParamHandler(e.target.value)); 
        if(this.uiElements.setNParamButton && this.uiElements.nParamInput)this.uiElements.setNParamButton.addEventListener('click',()=>{nParamHandler(this.uiElements.nParamInput.value); this.uiElements.nParamInput.blur();}); 
        if(this.uiElements.nParamInput)this.uiElements.nParamInput.addEventListener('change',(e)=>nParamHandler(e.target.value)); 
        
        if(this.uiElements.particleSpeedSlider)this.uiElements.particleSpeedSlider.addEventListener('input',(e)=>{
            const v = parseFloat(e.target.value);
            const MAX_SIM_SPEED = 100.0; 
            const MIN_SIM_SPEED = 0.1;
            const MID_SLIDER_VALUE = 50.0; 
            if (v <= MID_SLIDER_VALUE) { 
                this.particleSimulationSpeedScale = MIN_SIM_SPEED + (1.0 - MIN_SIM_SPEED) * (v / MID_SLIDER_VALUE);
            } else { 
                this.particleSimulationSpeedScale = 1.0 + (MAX_SIM_SPEED - 1.0) * ((v - MID_SLIDER_VALUE) / (100.0 - MID_SLIDER_VALUE));
            }
            this.particleSimulationSpeedScale = Math.max(MIN_SIM_SPEED, Math.min(this.particleSimulationSpeedScale, MAX_SIM_SPEED));
            if(this.uiElements.speedValueText)this.uiElements.speedValueText.textContent=`${this.particleSimulationSpeedScale.toFixed(2)}x`;
        }); 
        
        if(this.uiElements.toggleDesktopAudioButton)this.uiElements.toggleDesktopAudioButton.addEventListener('click',()=>this._toggleDesktopAudio());
        if(this.uiElements.toggleSoundButton)this.uiElements.toggleSoundButton.addEventListener('click',()=>this._toggleGeneratedSoundPlayback()); 
        if(this.uiElements.toggleFreezeButton)this.uiElements.toggleFreezeButton.addEventListener('click',()=>{this.areParticlesFrozen=!this.areParticlesFrozen; this._updateUIToggleButtons();}); 
        if(this.uiElements.toggleSubtitlesButton)this.uiElements.toggleSubtitlesButton.addEventListener('click',()=>{this.isSubtitlesEnabled = !this.isSubtitlesEnabled; if(this.subtitleContainer) this.subtitleContainer.classList.toggle('visible', this.isSubtitlesEnabled && this.currentSubtitles.length > 0 && this.subtitleContainer.textContent !== ''); this._updateUIToggleButtons();});
        if(this.uiElements.toggleFDMOptButton) {this.uiElements.toggleFDMOptButton.addEventListener('click', () => {this.enableFDMOptimization = !this.enableFDMOptimization; this._updateUIToggleButtons(); this._resetFullSimulationState(true);});}
        if(this.uiElements.toggleShadowsButton) {this.uiElements.toggleShadowsButton.addEventListener('click', () => {this.enableShadows = !this.enableShadows; this._applyShadowSettings(); this._updateUIToggleButtons();});}
        if(this.uiElements.toggleDynamicDensityButton) {this.uiElements.toggleDynamicDensityButton.addEventListener('click', () => {this.enableDynamicParticleDensity = !this.enableDynamicParticleDensity; if (!this.enableDynamicParticleDensity) {if (this.PARTICLE_COUNT !== this.MAX_PARTICLE_COUNT_USER_SETTING) {this.PARTICLE_COUNT = this.MAX_PARTICLE_COUNT_USER_SETTING; this._createParticleSystem();}} else { if(this.drivingMechanism === 'audio' || this.drivingMechanism === 'microphone') this.lastParticleCountUpdateTime = 0;} this._updateUIToggleButtons();});}
        if(this.uiElements.toggleMicrophoneButton) {this.uiElements.toggleMicrophoneButton.addEventListener('click', () => this._toggleMicrophoneInput());}
        if(this.uiElements.toggleStuckParticleCullingButton) {this.uiElements.toggleStuckParticleCullingButton.addEventListener('click', () => {this.enableStuckParticleCulling = !this.enableStuckParticleCulling; this._updateUIToggleButtons(); if(!this.enableStuckParticleCulling) this._resetParticlePositionsAndVelocities(); });}


        if(this.uiElements.resetSimulationButton)this.uiElements.resetSimulationButton.addEventListener('click',()=>this._resetAllSettingsToDefaults(false)); 
        if(this.uiElements.plateRotationSpeedSlider)this.uiElements.plateRotationSpeedSlider.addEventListener('input',(e)=>{this.plateRotationSpeed=parseFloat(e.target.value);if(this.uiElements.plateRotationSpeedValue)this.uiElements.plateRotationSpeedValue.textContent=`${this.plateRotationSpeed.toFixed(2)} об/сек`;}); 
        if(this.uiElements.stopRotationButton)this.uiElements.stopRotationButton.addEventListener('click',()=>{this.plateRotationSpeed=0;if(this.uiElements.plateRotationSpeedSlider)this.uiElements.plateRotationSpeedSlider.value=0;if(this.uiElements.plateRotationSpeedValue)this.uiElements.plateRotationSpeedValue.textContent="0.00 об/сек";}); 
        
        if(this.uiElements.audioFileInput)this.uiElements.audioFileInput.addEventListener('change',(e)=>this._handleAudioFileSelection(e)); 
        if(this.uiElements.playUploadedAudioButton)this.uiElements.playUploadedAudioButton.addEventListener('click',async ()=>{
            if (this.playlistFiles.length > 0) {
                const indexToPlay = this.currentPlaylistIndex === -1 ? 0 : this.currentPlaylistIndex;
                await this._loadAndPlayTrack(indexToPlay);
            }
        }); 
        if(this.uiElements.stopAudioButton)this.uiElements.stopAudioButton.addEventListener('click',()=>this._stopLoadedAudioFilePlayback(false));
        if(this.uiElements.toggleAudioPauseButton)this.uiElements.toggleAudioPauseButton.addEventListener('click',()=>this._togglePauseLoadedAudioFilePlayback()); 
        if(this.uiElements.nextTrackButton) this.uiElements.nextTrackButton.addEventListener('click', () => this._nextTrack());
        if(this.uiElements.prevTrackButton) this.uiElements.prevTrackButton.addEventListener('click', () => this._prevTrack());
        
        if (this.uiElements.audioProgressSlider) { this.uiElements.audioProgressSlider.addEventListener('input', (e) => { if (this.audioElement && (this.audioElement.duration || this.audioFileDuration) > 0) { const duration = this.audioElement.duration || this.audioFileDuration; const sliderValue = this._parseInputNumber(e.target.value, 0, false, 0, 100); const seekTime = (sliderValue / 100.0) * duration; this.audioPlaybackOffset = Math.max(0, Math.min(seekTime, duration)); this.audioElement.currentTime = this.audioPlaybackOffset; this.audioFileCurrentTime = this.audioPlaybackOffset;  if (!(this.isAudioFilePlaying && !this.isAudioFilePaused)) { this._updateAudioFileProgressControlsUI(); } } }); } 
        if(this.uiElements.pianoOctaveSelect)this.uiElements.pianoOctaveSelect.addEventListener('change',(e)=>{this.currentPianoOctave=parseInt(e.target.value); if(this.uiElements.pianoOctaveSelect) this.uiElements.pianoOctaveSelect.blur(); if((this.activePianoKeys.size>0||this.keyboardPressedKeys.size>0)&&this.drivingMechanism==='piano'){const lastNote = Array.from(this.keyboardPressedKeys).pop() || Array.from(this.activePianoKeys).pop(); if (lastNote) {this._handlePianoKeyPress(this.keyToNoteMapping[lastNote]||lastNote,false,this.keyboardPressedKeys.has(lastNote),false);this._handlePianoKeyPress(this.keyToNoteMapping[lastNote]||lastNote,true,this.keyboardPressedKeys.has(lastNote),false );} } }); 
        
        const leftPanelBtn = this.uiElements.toggleLeftPanelButton;
        const rightPanelBtn = this.uiElements.toggleRightPanelButton;
        
        if(leftPanelBtn && this.uiElements.controls) {
            leftPanelBtn.addEventListener('click', () => {
                const panel = this.uiElements.controls;
                panel.classList.toggle('hidden');
                leftPanelBtn.classList.toggle('hidden-state', panel.classList.contains('hidden'));
                const key = panel.classList.contains('hidden') ? 'toggle_left_panel_hide' : 'toggle_left_panel_show';
                if(LANG_PACK[this.currentLanguage]) leftPanelBtn.textContent = LANG_PACK[this.currentLanguage][key];
            });
        }
        
        if(rightPanelBtn && this.uiElements['advanced-controls']) {
            rightPanelBtn.addEventListener('click', () => {
                const panel = this.uiElements['advanced-controls'];
                panel.classList.toggle('hidden');
                rightPanelBtn.classList.toggle('hidden-state', panel.classList.contains('hidden'));
                const key = panel.classList.contains('hidden') ? 'toggle_right_panel_hide' : 'toggle_right_panel_show';
                if(LANG_PACK[this.currentLanguage]) rightPanelBtn.textContent = LANG_PACK[this.currentLanguage][key];
            });
        }

        this._setupAdvancedControlsListeners(); 
    }
    
    _setupUX() {
        this._setupLocalization();
        this._setupTooltips();
        this._setupModals();
    }
    
    _setupLocalization() {
        const savedLang = localStorage.getItem('chladni_lang');
        const browserLang = navigator.language.split('-')[0];
        const lang = savedLang || (LANG_PACK[browserLang] ? browserLang : 'ru');
        this._setLanguage(lang);
    }
    
    _setLanguage(lang) {
        if (!LANG_PACK[lang]) return;
        this.currentLanguage = lang;
        localStorage.setItem('chladni_lang', lang);

        const welcomeBody = this.uiElements['welcome-body-content'];
        if (welcomeBody && LANG_PACK[lang]['welcome_body']) {
            welcomeBody.innerHTML = LANG_PACK[lang]['welcome_body'];
        }
    
        document.querySelectorAll('[data-lang-key]').forEach(el => {
            const key = el.dataset.langKey;
            const translation = LANG_PACK[lang][key];
            
            if (translation && !el.dataset.langKey.startsWith('btn_')) {
                if ((el.tagName === 'LABEL' || el.tagName === 'P') && el.querySelector('span')) {
                    const textNode = Array.from(el.childNodes).find(node => node.nodeType === 3 && node.textContent.trim() !== '');
                    if (textNode) {
                        textNode.textContent = translation;
                    }
                } else if (!el.querySelector('div, span, p, label')) {
                    el.innerHTML = translation;
                }
            }
        });
        
        this._updateUIToggleButtons();
        this._updateModalParametersUI();

        const leftPanelBtn = this.uiElements.toggleLeftPanelButton;
        if (leftPanelBtn) {
            const key = leftPanelBtn.classList.contains('hidden-state') ? 'toggle_left_panel_hide' : 'toggle_left_panel_show';
            if (LANG_PACK[this.currentLanguage] && LANG_PACK[this.currentLanguage][key]) {
                 leftPanelBtn.textContent = LANG_PACK[this.currentLanguage][key];
            }
        }
        const rightPanelBtn = this.uiElements.toggleRightPanelButton;
        if (rightPanelBtn) {
            const key = rightPanelBtn.classList.contains('hidden-state') ? 'toggle_right_panel_hide' : 'toggle_right_panel_show';
            if (LANG_PACK[this.currentLanguage] && LANG_PACK[this.currentLanguage][key]) {
                rightPanelBtn.textContent = LANG_PACK[this.currentLanguage][key];
            }
        }
    }
    
    _setupTooltips() {
        const tooltipElement = this.uiElements.tooltip;
        if (!tooltipElement) return;
    
        document.querySelectorAll('.control-group[data-tooltip-key]').forEach(el => {
            el.addEventListener('mouseenter', (event) => {
                const key = el.dataset.tooltipKey;
                this._showTooltip(key, event.currentTarget);
            });
            el.addEventListener('mouseleave', () => {
                this._hideTooltip();
            });
        });
    }
    
    _showTooltip(key, targetElement) {
        if (this.tooltipTimeout) clearTimeout(this.tooltipTimeout);
        const tooltipData = TOOLTIP_TEXTS[key]?.[this.currentLanguage];
        if (!tooltipData) return;
    
        const tooltipElement = this.uiElements.tooltip;
        tooltipElement.innerHTML = `<h4>${tooltipData.title}</h4><p>${tooltipData.body}</p>`;
        
        const rect = targetElement.getBoundingClientRect();
        tooltipElement.classList.add('visible');
        const tooltipRect = tooltipElement.getBoundingClientRect();

        let x = rect.right + 15;
        let y = rect.top + (rect.height / 2) - (tooltipRect.height / 2);
        
        if (x + tooltipRect.width > window.innerWidth) {
            x = rect.left - tooltipRect.width - 15;
        }
        if (y + tooltipRect.height > window.innerHeight) {
            y = window.innerHeight - tooltipRect.height - 10;
        }
        if (y < 10) {
            y = 10;
        }

        tooltipElement.style.left = `${x}px`;
        tooltipElement.style.top = `${y}px`;
    }
    
    _hideTooltip() {
        this.tooltipTimeout = setTimeout(() => {
            this.uiElements.tooltip.classList.remove('visible');
        }, 100);
    }
    
    _setupModals() {
        const welcomeOverlay = this.uiElements['welcome-overlay'];
        const promptOverlay = this.uiElements['prompt-overlay'];
        
        const closeModalBtn = this.uiElements['close-modal-btn'];
        const showWelcomeBtn = this.uiElements['show-welcome-btn'];
        const showPromptBtn = this.uiElements['show-prompt-btn'];
        const closePromptBtn = this.uiElements['close-prompt-btn'];
        const langToggleBtn = this.uiElements['lang-toggle-btn'];
        const promptTextarea = this.uiElements['prompt-textarea'];
    
        const closeWelcome = () => {
            welcomeOverlay.classList.add('hidden');
            localStorage.setItem('chladni_visited', 'true');
        };
    
        const openWelcome = () => {
            welcomeOverlay.classList.remove('hidden');
        };
    
        if (!localStorage.getItem('chladni_visited')) {
            openWelcome();
        } else {
            welcomeOverlay.classList.add('hidden');
        }
    
        if(closeModalBtn) closeModalBtn.addEventListener('click', () => { closeWelcome(); this._startSpecialTrack(); });
        if(showWelcomeBtn) showWelcomeBtn.addEventListener('click', openWelcome);
    
        if(showPromptBtn) {
            showPromptBtn.addEventListener('click', () => {
                if(promptTextarea) promptTextarea.value = PROJECT_PROMPT_TEXT;
                if(welcomeOverlay) welcomeOverlay.classList.add('hidden');
                if(promptOverlay) promptOverlay.classList.remove('hidden');
            });
        }
    
        if(closePromptBtn) {
            closePromptBtn.addEventListener('click', () => {
                if(promptOverlay) promptOverlay.classList.add('hidden');
                openWelcome();
            });
        }
    
        if(langToggleBtn) {
            langToggleBtn.addEventListener('click', () => {
                const newLang = this.currentLanguage === 'ru' ? 'en' : 'ru';
                this._setLanguage(newLang);
            });
        }
    }
  
    async _startSpecialTrack() {
        const trackUrl = "https://dn721302.ca.archive.org/0/items/good-charlotte-the-click_202507/Good%20Charlotte%20-%20The%20Click.mp3";

        if (!trackUrl) {
            console.error("URL специального трека не указан.");
            return;
        }

        this.particleSimulationSpeedScale = 22.0;
        this.MAX_VISUAL_AMPLITUDE = 2.0;

        if (this.uiElements.particleSpeedSlider) {
            this.uiElements.particleSpeedSlider.value = 60.6;
            if (this.uiElements.speedValueText) this.uiElements.speedValueText.textContent = '22.00x';
        }
        if (this.uiElements.advMaxVisAmplitudeSlider) {
            this.uiElements.advMaxVisAmplitudeSlider.value = 2.0;
            if(this.uiElements.advMaxVisAmplitudeValue) this.uiElements.advMaxVisAmplitudeValue.textContent = '2.00';
        }

        try {
            if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = "Загрузка демонстрационного трека...";
            
            const response = await fetch(trackUrl);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            
            const blob = await response.blob();
            const audioFile = new File([blob], "Good Charlotte - The Click.mp3", { type: blob.type || 'audio/mpeg' });
            
            this.playlistFiles = [audioFile];
            this.currentPlaylistIndex = 0;
            await this._loadAndPlayTrack(0);

        } catch (error) {
            console.error("Не удалось загрузить специальный трек:", error);
            if (this.uiElements.audioInfoEl) this.uiElements.audioInfoEl.textContent = "Ошибка загрузки демо-трека.";
        }
    }

    _mainInitialization() {
        this._mapUIElements();
        this._setupUX();

        this._storeDefaultSimulationSettings();
        this._setupSubtitleSystem();
        this.MAX_PARTICLE_COUNT_USER_SETTING = PARTICLE_COUNT_DEFAULT;
        this.PARTICLE_COUNT = this.enableDynamicParticleDensity ? MIN_DYNAMIC_PARTICLE_COUNT : this.MAX_PARTICLE_COUNT_USER_SETTING;
        this.fdmConfiguredFrequency = this.actualAppliedFrequency;
        this.BPM_PEAK_DETECTION_WINDOW = BPM_PEAK_DETECTION_WINDOW_DEFAULT;
        this.smoothedPitchFrequency = this.currentFrequency;
        this.activeFetchID = 0;
        this.normalExcBaseAmp = this.EXCITATION_FREQ_DEP_BASE_AMP;
        this.besselZerosCache = {};
        this._setupThreeJSScene();
        this._setupWebAudioSystem();
        this._updatePhysicalConstants();
        this._createParticleSystem();
        this._createPianoKeys();
        this._setupEventListeners();
        this._populateAdvancedControlsUI();
        this._setActiveDrivingMechanism('modal');
        this._updateUIToggleButtons();
        this._applyShadowSettings();
        const gpuWarning = document.createElement('p'); 
        gpuWarning.textContent = 'FDM на GPU. Производительность значительно улучшена.'; 
        gpuWarning.style.color = '#5cb85c'; 
        gpuWarning.style.fontSize = '12px'; 
        gpuWarning.style.textAlign = 'center'; 
        gpuWarning.style.margin = '5px 0'; 
        gpuWarning.style.padding = '3px'; 
        gpuWarning.style.border = '1px solid #4cae4c'; 
        if (this.uiElements.controls) { 
            const fs = this.uiElements.controls.querySelector('fieldset'); 
            if (fs) this.uiElements.controls.insertBefore(gpuWarning, fs); 
            else this.uiElements.controls.appendChild(gpuWarning); 
        }
        this.animationClock.start();
        this._animateScene();
    }
    
    _handleWindowResize() {
        if (this.camera && this.renderer) {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        }
    }

    _animateScene() {
        requestAnimationFrame(this._animateScene.bind(this));
        const deltaTime = Math.min(this.animationClock.getDelta(), 0.05);
        if (this.enableDynamicParticleDensity) this._updateDynamicParticleDensity();
        if (this.isSubtitlesEnabled) this._updateSubtitles();
        if (this.orbitControls) this.orbitControls.update();
        if (this.plateRotationSpeed !== 0) {
            this.plateRotationAngle = (this.plateRotationAngle + this.plateRotationSpeed * 2 * Math.PI * deltaTime);
            if (this.plateRotationAngle > 2 * Math.PI) this.plateRotationAngle -= 2 * Math.PI;
            if (this.plateRotationAngle < 0) this.plateRotationAngle += 2 * Math.PI;
        }
        if (((this.drivingMechanism === 'audio' && this.isAudioFilePlaying && !this.isAudioFilePaused) ||
             (this.drivingMechanism === 'microphone' && this.isMicrophoneEnabled) ||
             (this.drivingMechanism === 'desktop_audio' && this.isDesktopAudioEnabled)
            ) && this.pitchDetectorAnalyserNode && this.mainAudioContext?.state === 'running') {
            const acTime = this.mainAudioContext.currentTime;
            if (acTime - this.lastPitchUpdateTime > PITCH_UPDATE_INTERVAL_SECONDS) {
                this.lastPitchUpdateTime = acTime;
                if (this.pitchDetectorSignalBuffer && this.pitchDetectorAnalyserNode) {
                    try {
                        this.pitchDetectorAnalyserNode.getFloatTimeDomainData(this.pitchDetectorSignalBuffer);
                        const detectedFreq = this._autoCorrelatePitch(this.pitchDetectorSignalBuffer, this.mainAudioContext.sampleRate);
                        if (detectedFreq !== -1) {
                            if (this.lastStablePitchFrequency === 0) this.lastStablePitchFrequency = detectedFreq;
                            this.lastStablePitchFrequency = this._linearInterpolate(this.lastStablePitchFrequency, detectedFreq, 0.3);
                            this.currentFrequency = detectedFreq;
                            if (this.uiElements.pitchDetectorInfo) {
                                this.uiElements.pitchDetectorInfo.style.display = 'block';
                                if (this.uiElements.pitch) this.uiElements.pitch.innerText = Math.round(detectedFreq);
                                let midi = this._midiNoteNumberFromFrequency(detectedFreq);
                                if (this.uiElements.note) this.uiElements.note.innerText = NOTE_NAMES_SHARP[midi % 12];
                                let cents = this._centsOffFromTruePitch(detectedFreq, midi);
                                if (this.uiElements.detune_amt) this.uiElements.detune_amt.innerText = Math.abs(cents);
                                if (this.uiElements.detune) this.uiElements.detune.className = cents === 0 ? "" : (cents < 0 ? "flat" : "sharp");
                            }
                            this.smoothedPitchFrequency = this._linearInterpolate(this.smoothedPitchFrequency, detectedFreq, PITCH_SMOOTHING_FACTOR);
                            if (Math.abs(this.fdmConfiguredFrequency - this.smoothedPitchFrequency) > PITCH_CHANGE_THRESHOLD_HZ) {
                                this.actualAppliedFrequency = this.smoothedPitchFrequency;
                                this._resetFullSimulationState(true);
                            } else {
                                this.actualAppliedFrequency = this.smoothedPitchFrequency;
                            }
                            this._updateFrequencyControlsUI();
                        } else if (this.uiElements.pitchDetectorInfo) {
                            if (this.drivingMechanism === 'audio' || this.drivingMechanism === 'microphone' || this.drivingMechanism === 'desktop_audio') {
                                if (this.uiElements.pitch) this.uiElements.pitch.innerText = '--';
                                if (this.uiElements.note) this.uiElements.note.innerText = '-';
                                if (this.uiElements.detune_amt) this.uiElements.detune_amt.innerText = '--';
                                if (this.uiElements.detune) this.uiElements.detune.className = '';
                            } else {
                                this.uiElements.pitchDetectorInfo.style.display = 'none';
                            }
                        }
                    } catch (e) { }
                }
            }
        }
        if (this.isAudioFilePlaying || this.isAudioFilePaused || this.audioElement?.controls) this._updateAudioFileProgressControlsUI();
        if (!this.areParticlesFrozen) {
            this._updatePlateFDM_GPU(deltaTime);
            this._updateParticles_GPU(deltaTime);
        } else if (this.plateRotationSpeed !== 0 && this.particlesMesh && this.particleData.length > 0) {
            const effectiveParticleCount = Math.round(this.PARTICLE_COUNT);
            if (this.particlesMesh.count !== effectiveParticleCount && effectiveParticleCount > 0) {
                 this._createParticleSystem();
                 this._resetParticlePositionsAndVelocities();
            }
            if (this.particlesMesh && this.particlesMesh.count === effectiveParticleCount && effectiveParticleCount > 0) {
                for (let i = 0; i < effectiveParticleCount; i++) {
                    const p = this.particleData[i];
                    if (!p || p.isHidden) continue;
                    const pos = p.position;
                    let h = 0;
                    if (this.u_curr_cpu_array2D) { const nX = (pos.x / this.plateWidth) + 0.5; const nY = (pos.y / this.plateHeight) + 0.5; const rF = nY * (this.currentGridSize - 1); const cF = nX * (this.currentGridSize - 1); const d = this._getDisplacementFromGPUField(rF, cF); h = THREE.MathUtils.clamp(d * this.VISUAL_DEFORMATION_SCALE, -this.MAX_VISUAL_AMPLITUDE, this.MAX_VISUAL_AMPLITUDE); }
                    const rx = pos.x * Math.cos(this.plateRotationAngle) - pos.y * Math.sin(this.plateRotationAngle);
                    const rz = pos.x * Math.sin(this.plateRotationAngle) + pos.y * Math.cos(this.plateRotationAngle);
                    this.particleInstanceMatrix.setPosition(rx, h, rz);
                    this.particlesMesh.setMatrixAt(i, this.particleInstanceMatrix);
                }
                if (this.particlesMesh) this.particlesMesh.instanceMatrix.needsUpdate = true;
            }
        }
        if (this.particlesMesh && this.particlesMesh.material.uniforms.globalColor) {
            this._updateParticleColorBasedOnFrequency();
        }
        if (this.renderer && this.scene && this.camera) this.renderer.render(this.scene, this.camera);
    }

    _parseLRC(lrcText) {
        if (!lrcText) return [];
        const lines = lrcText.split('\n');
        const subtitles = [];
        const timeRegex = /\[(\d{2}):(\d{2})\.(\d{2,3})\]/;
        lines.forEach(line => {
            const match = line.match(timeRegex);
            if (match) {
                const minutes = parseInt(match[1], 10);
                const seconds = parseInt(match[2], 10);
                const milliseconds = parseInt(match[3].padEnd(3, '0'), 10);
                const time = minutes * 60 + seconds + milliseconds / 1000;
                const text = line.replace(timeRegex, '').trim();
                if (text) {
                    subtitles.push({ time, text });
                }
            }
        });
        subtitles.sort((a, b) => a.time - b.time);
        return subtitles;
    }
    
    _parseTrackInfoFromName(fileName) {
        let cleanedName = fileName.replace(/\.[^/.]+$/, "").replace(/_/g, ' ');
        cleanedName = cleanedName.replace(/^♫\s*/, '');
        cleanedName = cleanedName.replace(/\s*\[.*?\]\s*/g, ' ').replace(/\s*\(.*?\)\s*/g, ' ').replace(/\s*\{.*?\}\s*/g, ' ').trim();
        cleanedName = cleanedName.replace(/^\s*\d+[\s.-]*/, '');
        const junkWords = ['official', 'video', 'lyrics', 'remix', 'live', 'acoustic', 'explicit', 'clean', 'audio', 'hq'];
        const junkRegex = new RegExp(`\\b(${junkWords.join('|')})\\b`, 'gi');
        cleanedName = cleanedName.replace(junkRegex, '').replace(/\s+/g, ' ').trim();
    
        const parts = cleanedName.split(' - ');
        let artist = '';
        let title = '';
    
        if (parts.length >= 2) {
            artist = parts[0].trim();
            title = parts.slice(1).join(' - ').trim();
        } else {
            title = cleanedName.trim();
        }
        
        return { artist, title };
    }
    
    _readID3Tags(file) {
        return new Promise((resolve, reject) => {
            window.jsmediatags.read(file, {
                onSuccess: (tag) => resolve(tag.tags),
                onError: (error) => reject(error)
            });
        });
    }

    async _findAndLoadLyrics(file, fetchID) {
        const lyricsEl = this.uiElements.lyricsInfoEl;
        lyricsEl.textContent = 'Поиск субтитров...';
        
        let artist, title;

        try {
            const tags = await this._readID3Tags(file);
            if (fetchID !== this.activeFetchID) return;

            const lyricsData = tags.lyrics || tags.USLT;
            if (lyricsData) {
                const lrcText = typeof lyricsData === 'string' ? lyricsData : lyricsData.lyrics;
                this.currentSubtitles = this._parseLRC(lrcText);
                if (this.currentSubtitles.length > 0) {
                    lyricsEl.textContent = 'Субтитры найдены (встроены).';
                    if (this.isSubtitlesEnabled) this.subtitleContainer.classList.add('visible');
                    return;
                }
            }
            artist = tags.artist?.trim();
            title = tags.title?.trim();
        } catch (error) {}

        if (fetchID !== this.activeFetchID) return;

        if (!artist || !title) {
            const infoFromName = this._parseTrackInfoFromName(file.name);
            artist = infoFromName.artist || artist;
            title = infoFromName.title || title;
        }

        if (title) {
            await this._fetchLyricsFromLrclib(artist || '', title, fetchID);
        } else {
            lyricsEl.textContent = 'Не удалось найти метаданные для поиска.';
        }
    }

    async _fetchLyricsFromLrclib(artist, track, requestIndex) {
        if (this.uiElements.lyricsInfoEl) {
            this.uiElements.lyricsInfoEl.textContent = 'Идет поиск текста в сети...';
        }
    
        const attempts = [
            { artist, track },
            { artist, track: track.replace(/\s*\(.*?\)\s*/g, '').trim() },
            { artist, track: track.toLowerCase() },
            { artist: artist.toLowerCase(), track: track.toLowerCase() }
        ];

        if (artist) {
            attempts.push({ artist: '', track });
        }
    
        for (const attempt of attempts) {
            if (this.activeFetchID !== requestIndex) return;
            if (!attempt.track) continue;

            const url = `https://lrclib.net/api/get?artist_name=${encodeURIComponent(attempt.artist)}&track_name=${encodeURIComponent(attempt.track)}`;
            
            try {
                const response = await fetch(url, { signal: AbortSignal.timeout(8000) });
                if (this.activeFetchID !== requestIndex) return;
                if (!response.ok) continue;

                const data = await response.json();
                if (data && data.syncedLyrics) {
                    this.currentSubtitles = this._parseLRC(data.syncedLyrics);
                    if (this.currentSubtitles.length > 0) {
                        if (this.uiElements.lyricsInfoEl) {
                            this.uiElements.lyricsInfoEl.textContent = 'Субтитры найдены (lrclib.net)';
                        }
                        if (this.isSubtitlesEnabled) this.subtitleContainer.classList.add('visible');
                        return;
                    }
                }
            } catch (e) {}
        }
    
        if (this.activeFetchID === requestIndex && this.uiElements.lyricsInfoEl) {
            this.uiElements.lyricsInfoEl.textContent = 'Субтитры не найдены.';
        }
    }

    _updateSubtitles() {
        if (!this.audioElement || this.currentSubtitles.length === 0) {
            if (this.subtitleContainer.textContent !== '') this.subtitleContainer.textContent = '';
            return;
        }
    
        const currentTime = this.audioElement.currentTime;
        let low = 0;
        let high = this.currentSubtitles.length - 1;
        let bestIndex = -1;
    
        while (low <= high) {
            const mid = Math.floor((low + high) / 2);
            if (this.currentSubtitles[mid].time <= currentTime) {
                bestIndex = mid;
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    
        const newText = (bestIndex !== -1) ? this.currentSubtitles[bestIndex].text : '';
        if (this.subtitleContainer.textContent !== newText) {
            this.subtitleContainer.textContent = newText;
            if (this.isSubtitlesEnabled) {
                this.subtitleContainer.classList.toggle('visible', newText !== '');
            }
        }
    }
}

async function main() {
    try {
        const response = await fetch('./data/bessel_roots.json');
        if (!response.ok) {
            throw new Error(`Не удалось загрузить bessel_roots.json: ${response.status} ${response.statusText}`);
        }
        const besselRootsData = await response.json();

        new ChladniSimulator(besselRootsData);

    } catch (error) {
        console.error("Критическая ошибка инициализации симулятора:", error);
        document.body.innerHTML = `<div style="color: #e06c75; background-color:#282c34; padding: 20px; font-family: sans-serif; height: 100vh; display: flex; align-items: center; justify-content: center; text-align: center;">
            <div>
                <h1>Критическая ошибка</h1>
                <p>Не удалось загрузить необходимые данные для симуляции.<br>
                   Убедитесь, что файл <b>/data/bessel_roots.json</b> существует и доступен.</p>
                <p style="font-family: monospace; background-color: #21252b; padding: 10px; border-radius: 5px;">${error.message}</p>
            </div>
        </div>`;
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main);
} else {
    main();
}
