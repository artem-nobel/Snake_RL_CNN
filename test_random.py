import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
#
#
# # def analyze_food_distribution(grid_width=20, grid_height=20, num_samples=10, spawn_range=(1, 18)):
# #     """
# #     Анализирует распределение генерации яблок
# #     """
# #     # Создаем массив для подсчета
# #     distribution = np.zeros((grid_height, grid_width), dtype=int)
# #
# #     # Симулируем генерацию яблок
# #     min_range, max_range = spawn_range
# #
# #     # Вариант 1A: Используем numpy random для лучшей равномерности
# #     x_coords = np.random.randint(min_range, max_range + 1, num_samples)
# #     y_coords = np.random.randint(min_range, max_range + 1, num_samples)
# #
# #     for x, y in zip(x_coords, y_coords):
# #         distribution[y, x] += 1
# #
# #     # Выводим статистику
# #     print(f"Всего генераций: {num_samples}")
# #     print(f"Диапазон генерации: от {min_range} до {max_range}")
# #     print(f"Размер поля: {grid_width}x{grid_height}")
# #     print(f"Плотность на клетку: {num_samples / ((max_range - min_range + 1) ** 2):.2f}")
# #     print("\nСтатистика распределения:")
# #     print(f"Максимальное значение в клетке: {distribution.max()}")
# #     print(f"Минимальное значение в клетке: {distribution.min()}")
# #     print(f"Среднее значение: {distribution.mean():.2f}")
# #     print(f"Стандартное отклонение: {distribution.std():.2f}")
# #
# #     # Анализ только области генерации
# #     gen_area = distribution[min_range:max_range + 1, min_range:max_range + 1]
# #     print(f"\nТолько область генерации ({gen_area.shape[0]}x{gen_area.shape[1]}):")
# #     print(f"Среднее в области: {gen_area.mean():.2f}")
# #     print(f"Мин в области: {gen_area.min()}")
# #     print(f"Макс в области: {gen_area.max()}")
# #     print(f"Неиспользованных клеток: {(gen_area == 0).sum()} из {gen_area.size}")
# #
# #     # Визуализация
# #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# #
# #     # 1. Heatmap распределения
# #     im1 = axes[0].imshow(distribution, cmap='hot', interpolation='nearest')
# #     axes[0].set_title('Heatmap распределения яблок')
# #     axes[0].set_xlabel('X координата')
# #     axes[0].set_ylabel('Y координата')
# #     plt.colorbar(im1, ax=axes[0], label='Количество генераций')
# #
# #     # 2. Гистограмма
# #     axes[1].hist(distribution[distribution > 0].flatten(), bins=50, edgecolor='black', alpha=0.7)
# #     axes[1].set_title('Гистограмма распределения')
# #     axes[1].set_xlabel('Количество генераций в клетке')
# #     axes[1].set_ylabel('Частота')
# #     axes[1].grid(True, alpha=0.3)
# #
# #     # 3. 3D поверхность
# #     from mpl_toolkits.mplot3d import Axes3D
# #     X, Y = np.meshgrid(range(grid_width), range(grid_height))
# #
# #     ax3d = fig.add_subplot(133, projection='3d')
# #     ax3d.plot_surface(X, Y, distribution, cmap='viridis', edgecolor='none', alpha=0.8)
# #     ax3d.set_title('3D поверхность распределения')
# #     ax3d.set_xlabel('X')
# #     ax3d.set_ylabel('Y')
# #     ax3d.set_zlabel('Количество')
# #
# #     plt.tight_layout()
# #     plt.show()
# #
# #     return distribution
# #
# #
# # # Анализ для вашего случая
# # print("=== Вариант 1: Numpy random ===")
# # distribution1 = analyze_food_distribution(
# #     grid_width=20,
# #     grid_height=20,
# #     num_samples=25,
# #     spawn_range=(1, 18)
# # )
#
#
# def analyze_food_distribution_shuffled(grid_width=20, grid_height=20, num_samples=1000, spawn_range=(1, 18)):
#     """
#     Анализирует распределение с предварительным перемешиванием
#     """
#     distribution = np.zeros((grid_height, grid_width), dtype=int)
#     min_range, max_range = spawn_range
#
#     # Создаем все возможные позиции
#     all_positions = [(x, y) for x in range(min_range, max_range + 1)
#                      for y in range(min_range, max_range + 1)]
#
#     # Перемешиваем один раз
#     random.shuffle(all_positions)
#
#     # Берем первые num_samples позиций (с циклическим повторением если нужно)
#     for i in range(num_samples):
#         x, y = all_positions[i % len(all_positions)]
#         distribution[y, x] += 1
#
#     # Выводим статистику
#     print(f"\n=== Вариант 2: Предварительное перемешивание ===")
#     print(f"Всего генераций: {num_samples}")
#     print(f"Доступных клеток: {len(all_positions)}")
#
#     # Анализ только области генерации
#     gen_area = distribution[min_range:max_range + 1, min_range:max_range + 1]
#     print(f"\nТолько область генерации:")
#     print(f"Среднее в области: {gen_area.mean():.2f}")
#     print(f"Мин в области: {gen_area.min()}")
#     print(f"Макс в области: {gen_area.max()}")
#     print(f"Разброс (макс/мин): {gen_area.max() / gen_area.min() if gen_area.min() > 0 else 'inf'}")
#
#     # Выводим статистику
#     print(f"Всего генераций: {num_samples}")
#     print(f"Диапазон генерации: от {min_range} до {max_range}")
#     print(f"Размер поля: {grid_width}x{grid_height}")
#     print(f"Плотность на клетку: {num_samples / ((max_range - min_range + 1) ** 2):.2f}")
#     print("\nСтатистика распределения:")
#     print(f"Максимальное значение в клетке: {distribution.max()}")
#     print(f"Минимальное значение в клетке: {distribution.min()}")
#     print(f"Среднее значение: {distribution.mean():.2f}")
#     print(f"Стандартное отклонение: {distribution.std():.2f}")
#
#     # Анализ только области генерации
#     gen_area = distribution[min_range:max_range + 1, min_range:max_range + 1]
#     print(f"\nТолько область генерации ({gen_area.shape[0]}x{gen_area.shape[1]}):")
#     print(f"Среднее в области: {gen_area.mean():.2f}")
#     print(f"Мин в области: {gen_area.min()}")
#     print(f"Макс в области: {gen_area.max()}")
#     print(f"Неиспользованных клеток: {(gen_area == 0).sum()} из {gen_area.size}")
#
#     # Визуализация
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
#     # 1. Heatmap распределения
#     im1 = axes[0].imshow(distribution, cmap='hot', interpolation='nearest')
#     axes[0].set_title('Heatmap распределения яблок')
#     axes[0].set_xlabel('X координата')
#     axes[0].set_ylabel('Y координата')
#     plt.colorbar(im1, ax=axes[0], label='Количество генераций')
#
#     # 2. Гистограмма
#     axes[1].hist(distribution[distribution > 0].flatten(), bins=50, edgecolor='black', alpha=0.7)
#     axes[1].set_title('Гистограмма распределения')
#     axes[1].set_xlabel('Количество генераций в клетке')
#     axes[1].set_ylabel('Частота')
#     axes[1].grid(True, alpha=0.3)
#
#     # 3. 3D поверхность
#     from mpl_toolkits.mplot3d import Axes3D
#     X, Y = np.meshgrid(range(grid_width), range(grid_height))
#
#     ax3d = fig.add_subplot(133, projection='3d')
#     ax3d.plot_surface(X, Y, distribution, cmap='viridis', edgecolor='none', alpha=0.8)
#     ax3d.set_title('3D поверхность распределения')
#     ax3d.set_xlabel('X')
#     ax3d.set_ylabel('Y')
#     ax3d.set_zlabel('Количество')
#
#     plt.tight_layout()
#     plt.show()
#
#     return distribution
#
#
#
# distribution2 = analyze_food_distribution_shuffled(
#     grid_width=20,
#     grid_height=20,
#     num_samples=30,
#     spawn_range=(1, 18)

# )


def analyze_food_distribution(grid_width=20, grid_height=20, num_samples=10, spawn_range=(1, 18)):
    """
    Анализирует распределение генерации яблок
    """
    # Создаем массив для подсчета
    distribution = np.zeros((grid_height, grid_width), dtype=int)

    # Симулируем генерацию яблок в 1000 эпизодов по 25 яблок
    min_range, max_range = spawn_range

    num_episodes = 1000
    apples_per_episode = 25

    print(f"=== Анализ для {num_episodes} эпизодов по {apples_per_episode} яблок ===")

    # ВНЕ цикла эпизодов: создаем ВСЕ возможные позиции один раз
    all_positions = [(x, y) for x in range(min_range, max_range + 1)
                     for y in range(min_range, max_range + 1)]

    # Перемешиваем все позиции один раз
    np.random.shuffle(all_positions)

    # Индекс для отслеживания текущей позиции в списке
    current_index = 0

    for episode in range(num_episodes):
        # В каждом эпизоде генерируем яблоки из общего буфера
        for _ in range(apples_per_episode):
            # Берем следующую позицию из перемешанного списка
            x, y = all_positions[current_index]
            current_index = (current_index + 1) % len(all_positions)

            # Если дошли до конца списка, перемешиваем заново
            if current_index == 0:
                np.random.shuffle(all_positions)

            distribution[y, x] += 1

    total_samples = num_episodes * apples_per_episode

    # Выводим статистику
    print(f"Всего генераций: {total_samples}")
    print(f"Диапазон генерации: от {min_range} до {max_range}")
    print(f"Размер поля: {grid_width}x{grid_height}")
    print(f"Плотность на клетку: {total_samples / ((max_range - min_range + 1) ** 2):.2f}")
    print("\nСтатистика распределения:")
    print(f"Максимальное значение в клетке: {distribution.max()}")
    print(f"Минимальное значение в клетке: {distribution.min()}")
    print(f"Среднее значение: {distribution.mean():.2f}")
    print(f"Стандартное отклонение: {distribution.std():.2f}")

    # Анализ только области генерации
    gen_area = distribution[min_range:max_range + 1, min_range:max_range + 1]
    print(f"\nТолько область генерации ({gen_area.shape[0]}x{gen_area.shape[1]}):")
    print(f"Среднее в области: {gen_area.mean():.2f}")
    print(f"Мин в области: {gen_area.min()}")
    print(f"Макс в области: {gen_area.max()}")
    print(f"Неиспользованных клеток: {(gen_area == 0).sum()} из {gen_area.size}")

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Heatmap распределения
    im1 = axes[0].imshow(distribution, cmap='hot', interpolation='nearest')
    axes[0].set_title('Heatmap распределения яблок (глобальный буфер)')
    axes[0].set_xlabel('X координата')
    axes[0].set_ylabel('Y координата')
    plt.colorbar(im1, ax=axes[0], label='Количество генераций')

    # 2. Гистограмма
    axes[1].hist(distribution[distribution > 0].flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_title('Гистограмма распределения')
    axes[1].set_xlabel('Количество генераций в клетке')
    axes[1].set_ylabel('Частота')
    axes[1].grid(True, alpha=0.3)

    # 3. 3D поверхность
    from mpl_toolkits.mplot3d import Axes3D
    X, Y = np.meshgrid(range(grid_width), range(grid_height))

    ax3d = fig.add_subplot(133, projection='3d')
    ax3d.plot_surface(X, Y, distribution, cmap='viridis', edgecolor='none', alpha=0.8)
    ax3d.set_title('3D поверхность распределения')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Количество')

    plt.tight_layout()
    plt.show()

    return distribution


# Анализ для вашего случая
print("=== Вариант: Глобальный буфер всех позиций ===")
distribution1 = analyze_food_distribution(
    grid_width=20,
    grid_height=20,
    num_samples=25,  # Этот параметр теперь не используется, но оставим для совместимости
    spawn_range=(1, 18)
)
# import random
# import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import colors


# def analyze_food_distribution(grid_width=20, grid_height=20, num_samples=10, spawn_range=(1, 18)):
#     """
#     Анализирует распределение генерации яблок
#     """
#     # Создаем массив для подсчета
#     distribution = np.zeros((grid_height, grid_width), dtype=int)
#
#     # Симулируем генерацию яблок в 1000 эпизодов по 25 яблок
#     min_range, max_range = spawn_range
#
#     num_episodes = 1000
#     apples_per_episode = 25
#
#     print(f"=== Анализ для {num_episodes} эпизодов по {apples_per_episode} яблок ===")
#
#     for episode in range(num_episodes):
#         # В каждом эпизоде генерируем яблоки
#         for _ in range(apples_per_episode):
#             x = np.random.randint(min_range, max_range + 1)
#             y = np.random.randint(min_range, max_range + 1)
#             distribution[y, x] += 1
#
#     total_samples = num_episodes * apples_per_episode
#
#     # Выводим статистику
#     print(f"Всего генераций: {total_samples}")
#     print(f"Диапазон генерации: от {min_range} до {max_range}")
#     print(f"Размер поля: {grid_width}x{grid_height}")
#     print(f"Плотность на клетку: {total_samples / ((max_range - min_range + 1) ** 2):.2f}")
#     print("\nСтатистика распределения:")
#     print(f"Максимальное значение в клетке: {distribution.max()}")
#     print(f"Минимальное значение в клетке: {distribution.min()}")
#     print(f"Среднее значение: {distribution.mean():.2f}")
#     print(f"Стандартное отклонение: {distribution.std():.2f}")
#
#     # Анализ только области генерации
#     gen_area = distribution[min_range:max_range + 1, min_range:max_range + 1]
#     print(f"\nТолько область генерации ({gen_area.shape[0]}x{gen_area.shape[1]}):")
#     print(f"Среднее в области: {gen_area.mean():.2f}")
#     print(f"Мин в области: {gen_area.min()}")
#     print(f"Макс в области: {gen_area.max()}")
#     print(f"Неиспользованных клеток: {(gen_area == 0).sum()} из {gen_area.size}")
#
#     # Визуализация
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#
#     # 1. Heatmap распределения
#     im1 = axes[0].imshow(distribution, cmap='hot', interpolation='nearest')
#     axes[0].set_title('Heatmap распределения яблок np.random.randint')
#     axes[0].set_xlabel('X координата')
#     axes[0].set_ylabel('Y координата')
#     plt.colorbar(im1, ax=axes[0], label='Количество генераций')
#
#     # 2. Гистограмма
#     axes[1].hist(distribution[distribution > 0].flatten(), bins=50, edgecolor='black', alpha=0.7)
#     axes[1].set_title('Гистограмма распределения')
#     axes[1].set_xlabel('Количество генераций в клетке')
#     axes[1].set_ylabel('Частота')
#     axes[1].grid(True, alpha=0.3)
#
#     # 3. 3D поверхность
#     from mpl_toolkits.mplot3d import Axes3D
#     X, Y = np.meshgrid(range(grid_width), range(grid_height))
#
#     ax3d = fig.add_subplot(133, projection='3d')
#     ax3d.plot_surface(X, Y, distribution, cmap='viridis', edgecolor='none', alpha=0.8)
#     ax3d.set_title('3D поверхность распределения')
#     ax3d.set_xlabel('X')
#     ax3d.set_ylabel('Y')
#     ax3d.set_zlabel('Количество')
#
#     plt.tight_layout()
#     plt.show()
#
#     return distribution
#
#
# # Анализ для вашего случая
# print("=== Вариант 1: Numpy random ===")
# distribution1 = analyze_food_distribution(
#     grid_width=20,
#     grid_height=20,
#     num_samples=25,  # Этот параметр теперь не используется, но оставим для совместимости
#     spawn_range=(1, 18)
# )


def analyze_food_distribution_shuffled(grid_width=20, grid_height=20, num_samples=1000, spawn_range=(1, 18)):
    """
    Анализирует распределение с предварительным перемешиванием
    """
    distribution = np.zeros((grid_height, grid_width), dtype=int)
    min_range, max_range = spawn_range

    # Параметры для эпизодного режима
    num_episodes = 1000
    apples_per_episode = 25

    print(f"\n=== Вариант 2: Предварительное перемешивание ===")
    print(f"Анализ для {num_episodes} эпизодов по {apples_per_episode} яблок")

    for episode in range(num_episodes):
        # На КАЖДОМ эпизоде создаем новый перемешанный список
        all_positions = [(x, y) for x in range(min_range, max_range + 1)
                         for y in range(min_range, max_range + 1)]

        # Перемешиваем один раз для этого эпизода
        random.shuffle(all_positions)

        # Берем apples_per_episode позиций для этого эпизода
        for i in range(apples_per_episode):
            x, y = all_positions[i % len(all_positions)]
            distribution[y, x] += 1

    total_samples = num_episodes * apples_per_episode

    # Выводим статистику
    print(f"Всего генераций: {total_samples}")
    print(f"Доступных клеток: {(max_range - min_range + 1) ** 2}")

    # Анализ только области генерации
    gen_area = distribution[min_range:max_range + 1, min_range:max_range + 1]
    print(f"\nТолько область генерации:")
    print(f"Среднее в области: {gen_area.mean():.2f}")
    print(f"Мин в области: {gen_area.min()}")
    print(f"Макс в области: {gen_area.max()}")
    print(f"Разброс (макс/мин): {gen_area.max() / gen_area.min() if gen_area.min() > 0 else 'inf'}")

    # Выводим статистику
    print(f"Всего генераций: {total_samples}")
    print(f"Диапазон генерации: от {min_range} до {max_range}")
    print(f"Размер поля: {grid_width}x{grid_height}")
    print(f"Плотность на клетку: {total_samples / ((max_range - min_range + 1) ** 2):.2f}")
    print("\nСтатистика распределения:")
    print(f"Максимальное значение в клетке: {distribution.max()}")
    print(f"Минимальное значение в клетке: {distribution.min()}")
    print(f"Среднее значение: {distribution.mean():.2f}")
    print(f"Стандартное отклонение: {distribution.std():.2f}")

    # Анализ только области генерации
    gen_area = distribution[min_range:max_range + 1, min_range:max_range + 1]
    print(f"\nТолько область генерации ({gen_area.shape[0]}x{gen_area.shape[1]}):")
    print(f"Среднее в области: {gen_area.mean():.2f}")
    print(f"Мин в области: {gen_area.min()}")
    print(f"Макс в области: {gen_area.max()}")
    print(f"Неиспользованных клеток: {(gen_area == 0).sum()} из {gen_area.size}")

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Heatmap распределения
    im1 = axes[0].imshow(distribution, cmap='hot', interpolation='nearest')
    axes[0].set_title('Heatmap распределения яблок random.shuffle')
    axes[0].set_xlabel('X координата')
    axes[0].set_ylabel('Y координата')
    plt.colorbar(im1, ax=axes[0], label='Количество генераций')

    # 2. Гистограмма
    axes[1].hist(distribution[distribution > 0].flatten(), bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_title('Гистограмма распределения')
    axes[1].set_xlabel('Количество генераций в клетке')
    axes[1].set_ylabel('Частота')
    axes[1].grid(True, alpha=0.3)

    # 3. 3D поверхность
    from mpl_toolkits.mplot3d import Axes3D
    X, Y = np.meshgrid(range(grid_width), range(grid_height))

    ax3d = fig.add_subplot(133, projection='3d')
    ax3d.plot_surface(X, Y, distribution, cmap='viridis', edgecolor='none', alpha=0.8)
    ax3d.set_title('3D поверхность распределения')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Количество')

    plt.tight_layout()
    plt.show()

    return distribution


distribution2 = analyze_food_distribution_shuffled(
    grid_width=20,
    grid_height=20,
    num_samples=30,  # Этот параметр теперь не используется, но оставим
    spawn_range=(1, 18)
)