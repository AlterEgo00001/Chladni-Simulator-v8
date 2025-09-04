import json
import numpy as np
from scipy.special import jn_zeros
import os
import time

# --- Параметры Генерации ---
# Мы устанавливаем эти значения, чтобы соответствовать данным, которые были
# изначально в вашем JS файле (m до 15, и около 10-15 корней на каждый порядок).
# Этого достаточно для всех пресетов и работы симулятора.
MAX_ORDER_M = 555       # Максимальный порядок m
ROOTS_PER_ORDER = 1001   # Количество корней n для каждого m
# ---------------------------

# Имя выходного файла и папки
OUTPUT_FOLDER = 'data'
OUTPUT_FILENAME = 'bessel_roots.json'

# Словарь для хранения финальных данных
bessel_roots = {}

print("="*50)
print("Начинаем вычисление корней функций Бесселя.")
print(f"Максимальный порядок (m): {MAX_ORDER_M}")
print(f"Корней на порядок (n): {ROOTS_PER_ORDER}")
print("="*50)

start_time = time.time()

# Основной цикл вычислений
for m in range(MAX_ORDER_M + 1):
    roots = jn_zeros(m, ROOTS_PER_ORDER)
    
    # Конвертируем m в строку для ключа JSON и массив NumPy в список Python
    bessel_roots[str(m)] = roots.tolist()
    print(f"  Порядок m={m}: вычислены корни.")

# Создаем папку 'data', если она не существует
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)

end_time_calc = time.time()
print("\n" + "="*50)
print(f"Вычисления завершены за {end_time_calc - start_time:.2f} секунд.")
print(f"Сохранение результатов в файл '{output_path}'...")

# Сохраняем словарь в файл в читаемом формате JSON
with open(output_path, "w", encoding='utf-8') as f:
    json.dump(bessel_roots, f, indent=4) # indent=4 для красивого вывода

print(f"Готово! Файл '{output_path}' успешно создан.")
print("="*50)