import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import tracemalloc
import sys


def generate_matrix(size, min_val=0, max_val=100):
    return np.random.randint(min_val, max_val + 1, size=(size, size, size), dtype=np.int32)


def linear_search(matrix, target):
    coords = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i, j, k] == target:
                    coords.append((i, j, k))
    return coords


def numpy_search(matrix, target):
    coords = np.argwhere(matrix == target)
    return[tuple(map(int, coord))for coord in coords]


def binary_search_3d(matrix, target):
    flat = np.sort(matrix.flatten())
    left, right = 0, len(flat) - 1
    found = False
    while left <= right:
        mid = (left + right) // 2
        if flat[mid] == target:
            found = True
            break
        elif flat[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    if not found:
        return []
    coords = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(matrix.shape[2]):
                if matrix[i, j, k] == target:
                    coords.append((i, j, k))
    return coords


def benchmark(func, *args):
    tracemalloc.start()
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, end - start, peak / 1024


def visualize(matrix, coords, target):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    all_x, all_y, all_z = np.where(matrix >= 0)
    ax.scatter(all_x, all_y, all_z, c='lightgray', alpha=0.3, s=10)
    if coords:
        x, y, z = zip(*coords)
        ax.scatter(x, y, z, c='red', s=80, edgecolors='darkred', linewidths=1.5)
        for i in range(len(x)):
            ax.text(x[i], y[i], z[i], f'({x[i]},{y[i]},{z[i]})', fontsize=8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Поиск значения {target} (найдено: {len(coords)})')
    plt.tight_layout()
    plt.show()


def test_numeral_systems(size=30, target=42):
    matrix = generate_matrix(size)
    systems = {
        'decimal': target,
        'binary': int(bin(target), 2),
        'hex': int(hex(target), 16),
        'octal': int(oct(target), 8)
    }
    results = {}
    for name, val in systems.items():
        _, t, mem = benchmark(numpy_search, matrix, val)
        results[name] = (t, mem)
    return results


def compare_algorithms(size=50, target=42):
    matrix = generate_matrix(size)
    results = {}
    for algo_name, algo_func in [('linear', linear_search), ('numpy', numpy_search), ('binary', binary_search_3d)]:
        coords, t, mem = benchmark(algo_func, matrix, target)
        results[algo_name] = {'time': t, 'memory_kb': mem, 'count': len(coords)}
    return results, matrix


if __name__ == '__main__':
    size = int(input('Размер матрицы (рекомендуется 20-50): ') or 30)
    target_input = input('Число для поиска (можно ввести 0b101010, 0x2A): ')
    try:
        target = int(eval(target_input))
    except:
        target = int(target_input)

    print(f'\nПоиск значения {target} в матрице {size}×{size}×{size}...')

    results, matrix = compare_algorithms(size, target)
    print('\nРезультаты бенчмарка:')
    for algo, data in results.items():
        print(
            f'{algo:8s} | время: {data["time"] * 1000:7.3f} мс | память: {data["memory_kb"]:7.2f} KB | найдено: {data["count"]}')

    coords = numpy_search(matrix, target)
    print(f'\nКоординаты найденных элементов: {coords if coords else "не найдено"}')

    if coords:
        visualize(matrix, coords, target)

    print('\nТест систем счисления (одинаковое значение 42):')
    sys_results = test_numeral_systems(size)
    for sys_name, (t, mem) in sys_results.items():
        print(f'{sys_name:8s} | время: {t * 1000:7.3f} мс | память: {mem:7.2f} KB')



























































































































