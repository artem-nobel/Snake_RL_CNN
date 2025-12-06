import os

import pygame
import torch
import random
import numpy as np
from collections import deque
import pygame
from numpy.ma.core import count

from snake_env import SnakeEnv
from modelVit import CNN_QNet, QTrainer
from helperVit import plot

MAX_MEMORY = 8096
BATCH_SIZE = 64
# LR = 0.001
LR = 0.00002

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 180  # 🚨 ИСПРАВЛЕНО: должно начинаться с 80 для exploration
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        #  параметры модели
        # input_channels=4 (4 кадра), num_actions=4 (вверх, вправо, вниз, влево)
        self.model = CNN_QNet(input_channels=4, num_actions=4)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    # метод наблюдения
    def get_state(self, env):

        return env._get_observation()  # Возвращает (4, 84, 84)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        # распаковка данных
        states, actions, rewards, next_states, dones = zip(*mini_sample)

        # конвертация в numpy массивы
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = list(actions)
        rewards = list(rewards)
        dones = list(dones)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # state уже должен быть numpy массивом (4, 84, 84)
        self.trainer.train_step(
            np.array([state], dtype=np.float32),  # Добавляем batch dimension
            [action],
            [reward],
            np.array([next_state], dtype=np.float32),
            [done]
        )

    def get_action(self, state):
        # логика epsilon-decay
        # epsilon уменьшается с 80 до 0 за 80 игр
        # count = self.n_games
        self.epsilon = max(0, 0 - self.n_games)
        # self.epsilon = max(0, 80 - (count / 10))
        final_move = [0, 0, 0, 0]  # 🚨 ИСПРАВЛЕНО: должно быть 4 действия!

        # Exploration
        if random.randint(0, 1300) < self.epsilon:  # 0-199 range
            move = random.randint(0, 3)  # 🚨 ИСПРАВЛЕНО: 0-3 (4 действия)
            final_move[move] = 1
        else:
            # Добавляем batch dimension для модели
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            # state_tensor shape: (1, 4, 84, 84)

            with torch.no_grad():
                prediction = self.model(state_tensor)

            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # 🚨 ИЗМЕНИТЕ render_mode с None на "human" или "rgb_array"
    agent = Agent()
    env = SnakeEnv(render_mode="human")  # 🟢 ИЗМЕНЕНО: было None, стало "human"
    # === ДОБАВЛЯЕМ ЗАГРУЗКУ ===
    load_path = "/Users/artemhorkov/desktop/RL/snake/model/model_game_3000.pth"
    if os.path.exists(load_path):
        print("🔄 Загружаю сохранённую модель...")
        agent.model.load_state_dict(torch.load(load_path))
        agent.n_games = 3000  # восстановить номер игры
        print("✅ Модель загружена, продолжаем обучение!")

    # Проверка формы состояния (опционально)
    test_state = agent.get_state(env)
    print(f"✅ State shape: {test_state.shape}")  # Должно быть (4, 84, 84)
    print(f"✅ State range: [{test_state.min():.3f}, {test_state.max():.3f}]")

    while True:
        # get old state
        state_old = agent.get_state(env)

        # get action
        final_move = agent.get_action(state_old)

        # perform step
        next_state, reward, done, ate, crashed = env.step(np.argmax(final_move))

        # 🟢 ДОБАВЛЕНО: вывод информации о шаге для отладки
        if agent.n_games % 10 == 0:  # Каждые 10 игр
            print(f"Game {agent.n_games}, Action: {np.argmax(final_move)}, Reward: {reward:.2f}, "
                      f"Score: {env.score}, Epsilon: {agent.epsilon}")

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, next_state, done)

        # remember
        agent.remember(state_old, final_move, reward, next_state, done)

        if done:
            score = env.score
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # agent.model.save()
                print(f"🎉 НОВЫЙ РЕКОРД! Сохранена модель с score: {score}")
            if agent.n_games % 500 == 0:
                save_path = f"/Users/artemhorkov/desktop/RL/snake/model/model_game_{agent.n_games}.pth"

                # Сохраняем полную модель (если нужно)
                torch.save(agent.model.state_dict(), save_path)

                # Сохраняем через тренер
                # self.model.save(save_path)


            print(f'Game {agent.n_games}, Score {score}, Record: {record}, Epsilon: {agent.epsilon}')

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # 🟢 ДОБАВЛЕНО: пауза для просмотра окна
            # pygame.time.delay(50)  # 0.5 секунды паузы между играми


if __name__ == '__main__':
    train()

