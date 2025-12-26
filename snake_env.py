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
        self.BASE_APPLE_REWARD = 10
        # self.APPLE_REWARD_MULTIPLIER = 1
        # self.EMPTY_STEP_PENALTY = -0.00
        self.DEATH_PENALTY = -100

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
            (start_x - 1, start_y),
            (start_x - 2, start_y)
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

    # def _spawn_food(self):
    #     while True:
    #         food = (random.randint(2, self.grid_width - 3),
    #                 random.randint(2, self.grid_height - 3))
    #
    #         if food not in self.snake:
    #             return food
    def _spawn_food(self):
        """
        Генерация яблока с равномерным распределением по полю.
        Все клетки используются по очереди, что предотвращает кластеризацию.
        """
        # Создаем все возможные позиции яблок (если еще не созданы)
        if not hasattr(self, 'all_food_positions'):
            self.food_min_x = 2
            self.food_max_x = self.grid_width - 3
            self.food_min_y = 2
            self.food_max_y = self.grid_height - 3

            self.all_food_positions = [
                (x, y)
                for x in range(self.food_min_x, self.food_max_x + 1)
                for y in range(self.food_min_y, self.food_max_y + 1)
            ]
            random.shuffle(self.all_food_positions)
            self.current_food_index = 0

        # Перебираем перемешанные позиции, пока не найдем свободную
        for _ in range(len(self.all_food_positions)):
            food = self.all_food_positions[self.current_food_index]
            self.current_food_index = (self.current_food_index + 1) % len(self.all_food_positions)

            # Если дошли до начала списка, перемешиваем заново
            if self.current_food_index == 0:
                random.shuffle(self.all_food_positions)

            # Проверяем, что позиция не занята змеей
            if food not in self.snake:
                return food

        # Если все позиции заняты (крайне маловероятно), используем случайную
        while True:
            food = (random.randint(self.food_min_x, self.food_max_x),
                    random.randint(self.food_min_y, self.food_max_y))
            if food not in self.snake:
                return food
    # def _get_directional_reward(self):
    #     head_x, head_y = self.snake[0]
    #     food_x, food_y = self.food
    #
    #     old_dist = abs(head_x - food_x) + abs(head_y - food_y)
    #     new_head_x = head_x + self.direction[0]
    #     new_head_y = head_y + self.direction[1]
    #     new_dist = abs(new_head_x - food_x) + abs(new_head_y - food_y)
    #
    #     return 0.1 if new_dist < old_dist else -0.1

    # def _calculate_directional_reward(self, old_x, old_y, new_x, new_y):
    #     """Вычисляет награду за движение к/от еды"""
    #     food_x, food_y = self.food

        # # Манхэттенское расстояние (L1 норма)
        # old_distance = abs(old_x - food_x) + abs(old_y - food_y)
        # new_distance = abs(new_x - food_x) + abs(new_y - food_y)
        #
        # if new_distance < old_distance:
        #     return 0.0  # 🟢 Награда за движение К еде
        # elif new_distance > old_distance:
        #     return -0.0  # 🔴 Штраф за движение ОТ еды
        # else:
        #     return 0.0  # 🤷 Без изменений (движение параллельно)

    def step(self, action):
        self.ate_apple_this_step = False
        self.crashed_this_step = False

        self._update_direction(action)

        head_x, head_y = self.snake[0]
        new_head_x = head_x + self.direction[0]
        new_head_y = head_y + self.direction[1]
        new_head = (new_head_x, new_head_y)

        # directional_reward = self._calculate_directional_reward(head_x, head_y, new_head_x, new_head_y)

        # reward = directional_reward
        reward = 0
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
        frame_array = np.transpose(frame_array, (1, 0, 2))  # из (W,H,C) в (H,W,C)

        return self._preprocess_frame_cv(frame_array)

    # def _get_observation(self):
    #     return np.stack(self.frames, axis=0)
    def _get_observation(self):
        """Возвращает стек обработанных кадров [4, 84, 84]"""

        return np.stack(self.frames, axis=0)

    def close(self):
        pygame.quit()
