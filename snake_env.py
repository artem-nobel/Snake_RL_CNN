import pygame
import numpy as np
import random
from collections import deque
import cv2  # 🆕 OpenCV


class SnakeEnv:
    def __init__(self, render_mode=None):
        self.screen_width = 400
        self.screen_height = 400
        self.cell_size = 20
        self.grid_width = self.screen_width // self.cell_size
        self.grid_height = self.screen_height // self.cell_size

        self.frame_stack = 4
        self.frames = deque(maxlen=self.frame_stack)

        # Rewards
        self.BASE_APPLE_REWARD = 20
        # self.APPLE_REWARD_MULTIPLIER = 1
        # self.EMPTY_STEP_PENALTY = -0.00
        self.DEATH_PENALTY = -20

        self.render_mode = render_mode
        self.display_screen = None

        pygame.init()
        self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.render_mode == "human":
            self.display_screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Snake AI ")

        self.BLACK = (0, 0, 0)
        self.DARK_GRAY = (64, 64, 64)
        self.LIGHT_GRAY = (192, 192, 192)
        self.WHITE = (255, 255, 255)

        self.reset()

    def reset(self):
        start_x = self.grid_width // 2
        start_y = self.grid_height // 2

        self.snake = [
            (start_x, start_y),
            (start_x - 1, start_y)
        ]

        self.direction = (1, 0)
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.empty_steps = 0
        self.consecutive_apples = 0
        self.game_over = False
        self.ate_apple_this_step = False
        self.crashed_this_step = False

        self.frames.clear()

        initial_frame = self._get_processed_frame()
        for _ in range(self.frame_stack):
            self.frames.append(initial_frame)

        return self._get_observation()

    def _spawn_food(self):
        while True:
            food = (random.randint(2, self.grid_width - 3),
                    random.randint(2, self.grid_height - 3))

            if food not in self.snake:
                return food

    def _get_directional_reward(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        old_dist = abs(head_x - food_x) + abs(head_y - food_y)
        new_head_x = head_x + self.direction[0]
        new_head_y = head_y + self.direction[1]
        new_dist = abs(new_head_x - food_x) + abs(new_head_y - food_y)

        return 0.1 if new_dist < old_dist else -0.1

    def _calculate_directional_reward(self, old_x, old_y, new_x, new_y):
        """Вычисляет награду за движение к/от еды"""
        food_x, food_y = self.food

        # Манхэттенское расстояние (L1 норма)
        old_distance = abs(old_x - food_x) + abs(old_y - food_y)
        new_distance = abs(new_x - food_x) + abs(new_y - food_y)

        if new_distance < old_distance:
            return 1  # 🟢 Награда за движение К еде
        elif new_distance > old_distance:
            return -1  # 🔴 Штраф за движение ОТ еды
        else:
            return 0.0  # 🤷 Без изменений (движение параллельно)

    def step(self, action):
        self.ate_apple_this_step = False
        self.crashed_this_step = False

        self._update_direction(action)

        head_x, head_y = self.snake[0]
        new_head_x = head_x + self.direction[0]
        new_head_y = head_y + self.direction[1]
        new_head = (new_head_x, new_head_y)

        directional_reward = self._calculate_directional_reward(head_x, head_y, new_head_x, new_head_y)

        reward = directional_reward
        # self.score += directional_reward
        # reward = self.EMPTY_STEP_PENALTY
        self.game_over = False

        # Столкновение со стенами
        if (new_head_x < 0 or new_head_x >= self.grid_width or
                new_head_y < 0 or new_head_y >= self.grid_height):
            self.game_over = True
            reward = self.DEATH_PENALTY
            self.crashed_this_step = True
            return self._get_observation(), reward, self.game_over, self.ate_apple_this_step, self.crashed_this_step

        # Столкновение с собой
        if new_head in self.snake:
            self.game_over = True
            reward = self.DEATH_PENALTY
            self.crashed_this_step = True
        else:
            self.snake.insert(0, new_head)

            if new_head == self.food:
                self.score += 1
                self.consecutive_apples += 1
                reward = self.BASE_APPLE_REWARD
                self.food = self._spawn_food()
                self.ate_apple_this_step = True
                self.empty_steps = 0
            else:
                self.snake.pop()
                self.empty_steps += 1

                # if self.empty_steps >= self.MAX_STEPS_WITHOUT_APPLE:
                #     self.game_over = True
                #     reward = 0 + directional_reward
                #     self.crashed_this_step = True

        self.steps += 1

        if self.render_mode == "human":
            self._render_frame()

        new_frame = self._get_processed_frame()
        self.frames.append(new_frame)

        return self._get_observation(), reward, self.game_over, self.ate_apple_this_step, self.crashed_this_step

    def _update_direction(self, action):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        new_direction = directions[action]

        if (new_direction[0] != -self.direction[0] or
                new_direction[1] != -self.direction[1]):
            self.direction = new_direction

    def render(self):
        self.screen.fill(self.BLACK)

        for i, (x, y) in enumerate(self.snake):
            color = self.LIGHT_GRAY if i == 0 else self.DARK_GRAY
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
                               self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, self.BLACK, rect, 1)

        food_rect = pygame.Rect(self.food[0] * self.cell_size,
                                self.food[1] * self.cell_size,
                                self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self.WHITE, food_rect)

        font = pygame.font.SysFont('Arial', 16)
        score_text = font.render(f'Score: {self.score}', True, self.WHITE)
        self.screen.blit(score_text, (10, 10))

        return self.screen

    def _render_frame(self):
        game_surface = self.render()
        if self.display_screen:
            self.display_screen.blit(game_surface, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            pygame.time.delay(50)

    def _preprocess_frame_cv(self, frame):
        """Grayscale + resize через OpenCV (быстро и стабильно)"""
        # 🚨 ИСПРАВЛЕНО: frame уже в формате (C, H, W), нужно переставить оси
        # Конвертируем из (C, H, W) в (H, W, C) для OpenCV
        if frame.shape[0] == 3:  # Если каналы первыми
            frame = np.transpose(frame, (1, 2, 0))

        # 1. Конвертация RGB → grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 2. Resize до 84×84
        img_resized = cv2.resize(img_gray, (84, 84), interpolation=cv2.INTER_AREA)

        # 3. Нормализация
        img_norm = img_resized.astype(np.float32) / 255.0

        return img_norm
    # =====================================================================
    # Замена _get_processed_frame на OpenCV
    # =====================================================================
    def _get_processed_frame(self):
        frame_surface = self.render()

        #  через OpenCV
        frame_array = pygame.surfarray.array3d(frame_surface)
        frame_array = np.transpose(frame_array, (2, 0, 1))  # из (W,H,C) в (H,W,C)

        return self._preprocess_frame_cv(frame_array)

    # def _get_observation(self):
    #     return np.stack(self.frames, axis=0)
    def _get_observation(self):
        """Возвращает стек обработанных кадров [4, 84, 84]"""

        return np.stack(self.frames, axis=0)

    def close(self):
        pygame.quit()











# import pygame
# import numpy as np
# import random
# from collections import deque
#
#
# class SnakeEnv:
#     def __init__(self, render_mode=None):
#         self.screen_width = 400
#         self.screen_height = 400
#         self.cell_size = 20
#         self.grid_width = self.screen_width // self.cell_size
#         self.grid_height = self.screen_height // self.cell_size
#
#         # Настройки кадров для CNN
#         self.frame_stack = 4
#         self.frames = deque(maxlen=self.frame_stack)
#
#         # Настройки наград
#         self.BASE_APPLE_REWARD = 2
#         self.APPLE_REWARD_MULTIPLIER = 1
#         self.EMPTY_STEP_PENALTY = -0.00
#         self.DEATH_PENALTY = -1
#
#         # 🆕 Максимальное количество шагов без яблока
#         self.MAX_STEPS_WITHOUT_APPLE = 150
#
#
#         # Режим рендеринга
#         self.render_mode = render_mode
#         self.display_screen = None
#
#         # Инициализация Pygame
#         pygame.init()
#         self.screen = pygame.Surface((self.screen_width, self.screen_height))
#
#         # Если включен режим отображения, создаем окно
#         if self.render_mode == "human":
#             self.display_screen = pygame.display.set_mode((self.screen_width, self.screen_height))
#             pygame.display.set_caption("Snake AI ")
#
#         # Цвета
#         self.BLACK = (0, 0, 0) # фон
#         self.DARK_GRAY = (64, 64, 64)  # тело змейки
#         self.LIGHT_GRAY = (192, 192, 192)  # голова
#         self.WHITE = (255, 255, 255)  # еда
#
#
#         self.reset()
#
#     def reset(self):
#         """Сброс игры в начальное состояние"""
#         # Начальная длина 4
#         start_x = self.grid_width // 2
#         start_y = self.grid_height // 2
#         self.snake = [
#             (start_x, start_y),
#             (start_x - 1, start_y),
#             # (start_x - 2, start_y),
#             # (start_x - 3, start_y)
#         ]
#
#         self.direction = (1, 0)
#         self.food = self._spawn_food()
#         self.score = 0
#         self.steps = 0
#         self.empty_steps = 0
#         self.consecutive_apples = 0
#         self.game_over = False
#         self.ate_apple_this_step = False
#         self.crashed_this_step = False  # 🆕 Флаг столкновения на этом шаге
#
#         # Очищаем историю кадров
#         self.frames.clear()
#         initial_frame = self._get_processed_frame()
#         for _ in range(self.frame_stack):
#             self.frames.append(initial_frame)
#
#         return self._get_observation()
#
#     def _spawn_food(self):
#         """Создает еду в случайной позиции, но не на змейке"""
#         while True:
#             # аблоко появляется везде
#             # food = (random.randint(0, self.grid_width - 1),
#             #         random.randint(0, self.grid_height - 1))
#
#             # 🚫 Не создавать еду на самом краю (1 клетка от границы)
#             # food = (random.randint(1, self.grid_width - 2),  # от 1 до width-2
#             #         random.randint(1, self.grid_height - 2))  # от 1 до height-2
#
#             # 🚫 аблоко появляется в 2 клетках от края:
#             food = (random.randint(2, self.grid_width - 3),   # от 2 до width-3
#                     random.randint(2, self.grid_height - 3))  # от 2 до height-3
#
#             if food not in self.snake:
#                 return food
#
#
#     def _get_directional_reward(self):
#         """Награда за движение к еде"""
#         head_x, head_y = self.snake[0]
#         food_x, food_y = self.food
#
#         old_dist = abs(head_x - food_x) + abs(head_y - food_y)
#         new_head_x = head_x + self.direction[0]
#         new_head_y = head_y + self.direction[1]
#         new_dist = abs(new_head_x - food_x) + abs(new_head_y - food_y)
#
#         return 0.1 if new_dist < old_dist else -0.1
#     def step(self, action):
#         """Выполняет действие и возвращает новое состояние, награды, done"""
#         # Сбрасываем флаги
#         self.ate_apple_this_step = False
#         self.crashed_this_step = False
#
#         # Конвертируем действие в направление
#         self._update_direction(action)
#
#         # Двигаем змейку ОБЫЧНЫМ способом (без телепортации)
#         head_x, head_y = self.snake[0]
#         new_head_x = head_x + self.direction[0]  # ← Обычное движение
#         new_head_y = head_y + self.direction[1]  # ← Обычное движение
#         new_head = (new_head_x, new_head_y)
#
#         # Проверяем столкновения
#         reward = self.EMPTY_STEP_PENALTY
#         self.game_over = False
#
#         # СНАЧАЛА проверяем столкновение со стенами
#         if (new_head_x < 0 or new_head_x >= self.grid_width or
#                 new_head_y < 0 or new_head_y >= self.grid_height):
#             self.game_over = True
#             reward = self.DEATH_PENALTY
#             self.consecutive_apples = 0
#             self.crashed_this_step = True
#             # Возвращаем состояние сразу
#             return self._get_observation(), reward, self.game_over, self.ate_apple_this_step, self.crashed_this_step
#
#         # Затем проверяем столкновение с собой
#         if new_head in self.snake:
#             self.game_over = True
#             reward = self.DEATH_PENALTY
#             self.consecutive_apples = 0
#             self.crashed_this_step = True
#
#         else:
#             # Добавляем новую голову
#             self.snake.insert(0, new_head)
#
#             # Проверяем, съели ли еду
#             if new_head == self.food:
#                 self.score += 1
#                 self.consecutive_apples += 1
#                 reward = self.BASE_APPLE_REWARD
#                 self.food = self._spawn_food()
#                 self.ate_apple_this_step = True
#                 self.empty_steps = 0
#             else:
#                 # Убираем хвост, если не съели еду
#                 self.snake.pop()
#                 self.empty_steps += 1
#
#                 if self.empty_steps >= self.MAX_STEPS_WITHOUT_APPLE:
#                     self.game_over = True
#                     reward = -0
#                     self.crashed_this_step = True
#
#         self.steps += 1
#
#         # Если включен рендеринг, отображаем
#         if self.render_mode == "human":
#             self._render_frame()
#
#         # Получаем новый кадр
#         new_frame = self._get_processed_frame()
#         self.frames.append(new_frame)
#
#         return self._get_observation(), reward, self.game_over, self.ate_apple_this_step, self.crashed_this_step
#
#
#     def _update_direction(self, action):
#         """Обновляет направление змейки на основе действия"""
#         directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
#         new_direction = directions[action]
#
#         # Не позволяем змейке развернуться на 180 градусов
#         if (new_direction[0] != -self.direction[0] or
#                 new_direction[1] != -self.direction[1]):
#             self.direction = new_direction
#
#     def render(self):
#         """Отрисовывает текущее состояние игры на Surface"""
#         self.screen.fill(self.BLACK)
#
#         # Рисуем змейку
#         for i, (x, y) in enumerate(self.snake):
#             color = self.LIGHT_GRAY if i == 0 else self.DARK_GRAY
#
#             rect = pygame.Rect(x * self.cell_size, y * self.cell_size,
#                                self.cell_size, self.cell_size)
#             pygame.draw.rect(self.screen, color, rect)
#             pygame.draw.rect(self.screen, self.BLACK, rect, 1)
#
#         # Рисуем еду
#         food_rect = pygame.Rect(self.food[0] * self.cell_size,
#                                 self.food[1] * self.cell_size,
#                                 self.cell_size, self.cell_size)
#         pygame.draw.rect(self.screen, self.WHITE, food_rect)
#
#
#         # Рисуем информацию
#         font = pygame.font.SysFont('Arial', 16)
#         score_text = font.render(f'Score: {self.score}', True, self.WHITE)
#         # empty_text = font.render(f'Empty: {self.empty_steps}/{self.MAX_STEPS_WITHOUT_APPLE}', True, self.WHITE)
#         # consecutive_text = font.render(f'Consecutive: {self.consecutive_apples}', True, self.WHITE)
#         # length_text = font.render(f'Length: {len(self.snake)}', True, self.WHITE)
#         # learn_text = font.render('Learn from: Apples & Crashes', True, self.WHITE)
#
#         # 🆕 Информация об обучении на столкновениях
#         # learn_text = font.render('Learn from: Apples & Crashes', True, self.PURPLE)
#
#         self.screen.blit(score_text, (10, 10))
#         # self.screen.blit(steps_text, (10, 30))
#         # self.screen.blit(empty_text, (10, 50))
#         # self.screen.blit(consecutive_text, (10, 70))
#         # self.screen.blit(length_text, (10, 90))
#         # self.screen.blit(reward_text, (10, 110))
#         # self.screen.blit(learn_text, (10, 130))  # 🆕
#
#         return self.screen
#
#     def _render_frame(self):
#         """Отображает кадр в окне Pygame"""
#         game_surface = self.render()
#
#         if self.display_screen:
#             self.display_screen.blit(game_surface, (0, 0))
#             pygame.display.flip()
#
#             # Обрабатываем события
#             for event in pygame.event.get():
#                 if event.type == pygame.QUIT:
#                     self.close()
#                     return
#
#             pygame.time.delay(50)
#
#     def _get_processed_frame(self):
#         """Возвращает предобработанный кадр для нейросети"""
#         frame_surface = self.render()
#         frame_array = pygame.surfarray.array3d(frame_surface)
#         frame_array = np.transpose(frame_array, (1, 0, 2))
#         return self._preprocess_frame(frame_array)
#
#     def _preprocess_frame(self, frame):
#         """Предобработка кадра: уменьшение размера"""
#         #сцена черно-белая, берем один канал
#         temp_surface = pygame.surfarray.make_surface(frame)
#         small_surface = pygame.transform.scale(temp_surface, (84, 84))
#         small_array = pygame.surfarray.array3d(small_surface)
#         frame_gray = np.dot(small_array[..., :3], [0.2989, 0.5870, 0.1140])
#
#         # Берем только красный канал (все каналы одинаковы в grayscale)
#         # frame_gray = small_array[..., 0]  # или [..., 1] или [..., 2] - все одинаковы
#         frame_gray = frame_gray.astype(np.float32) / 255.0
#         return frame_gray
#
#     def _get_observation(self):
#         """Возвращает stacked frames как tensor [4, 84, 84]"""
#         return np.stack(self.frames, axis=0)
#
#     def close(self):
#         pygame.quit()