import os
import torch
import random
import numpy as np

from collections import deque
from snake_env import SnakeEnv
from model import DuelingCNN_QNet, QTrainer
from helper import plot

MAX_MEMORY = 32384
BATCH_SIZE = 128
LR = 0.0001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 700
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)

        # DDQN: Добавляем счетчик шагов для периодического обновления
        self.learn_step_counter = 0
        self.target_update_freq = 1000

        # === DUELING: Меняем модель на DuelingCNN_QNet ===
        self.model = DuelingCNN_QNet(input_channels=4, num_actions=4)  # Изменено!
        # === DUELING: QTrainer теперь работает с Dueling сетью ===
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, tau=0.005)

    # метод наблюдения
    def get_state(self, env):
        return env._get_observation()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = list(actions)
        rewards = list(rewards)
        dones = list(dones)

        # Обучаем и получаем loss
        loss = self.trainer.train_step(states, actions, rewards, next_states, dones)

        self.learn_step_counter += 1

        if self.learn_step_counter % self.target_update_freq == 0:
            self.trainer.hard_update()
            print(f"🔄 Таргетная сеть полностью обновлена (шаг {self.learn_step_counter})")
            # === DUELING: Добавим информацию о типе сети ===
            print(f"📊 Loss: {loss:.4f}, Games: {self.n_games}, Network: Dueling DDQN")

        return loss

    def train_short_memory(self, state, action, reward, next_state, done):
        loss = self.trainer.train_step(
            np.array([state], dtype=np.float32),
            [action],
            [reward],
            np.array([next_state], dtype=np.float32),
            [done]
        )

        self.learn_step_counter += 1

        if self.learn_step_counter % self.target_update_freq == 0:
            self.trainer.hard_update()
            print(f"🔄 Таргетная сеть полностью обновлена (шаг {self.learn_step_counter})")

        return loss

    def get_action(self, state):
        final_move = [0, 0, 0, 0]
        # === DUELING: epsilon-decay ===
        # self.epsilon = max(0, self.epsilon - 1)  # Пример decay
        self.epsilon = max(0, 700 - self.n_games)  # Линейный decay

        # Exploration
        if random.randint(0, 1300) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            # DUELING
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                prediction = self.model(state_tensor)

            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():

    LOAD_MODEL = True  # True - загрузить модель и данные, False - начать с нуля
    # LOAD_MODEL = False
    # MODEL_TO_LOAD = None  # None - загрузить последнюю, или имя файла например "model_dueling_500.pth"
    MODEL_TO_LOAD = "model_dueling_2500.pth"
    # =============================================

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    loaded_n_games = 0

    if LOAD_MODEL:
        if MODEL_TO_LOAD is None:
            # Ищем последнюю сохраненную модель
            model_files = [f for f in os.listdir() if f.startswith("model_dueling_") and f.endswith(".pth")]
            if model_files:
                # Сортируем по номеру игры
                def extract_game_num(filename):
                    try:
                        return int(filename.split("_")[2].split(".")[0])
                    except:
                        return 0

                latest_model = max(model_files, key=extract_game_num)
                model_to_load = latest_model
            else:
                print("🚀 Модели не найдены, начинаем с нуля")
                LOAD_MODEL = False
        else:
            model_to_load = MODEL_TO_LOAD

        if LOAD_MODEL:
            # Проверяем существует ли файл модели
            if not os.path.exists(model_to_load):
                print(f"⚠️ Файл модели '{model_to_load}' не найден")
                print("🚀 Начинаем с нуля")
                LOAD_MODEL = False
            else:
                # Извлекаем номер игры из имени файла
                try:
                    game_num = int(model_to_load.split("_")[2].split(".")[0])
                except:
                    print(f"⚠️ Не могу извлечь номер игры из имени файла: {model_to_load}")
                    game_num = 0

                # Загружаем модель
                agent = Agent()
                agent.model.load_state_dict(torch.load(model_to_load))
                agent.trainer.hard_update()

                # Загружаем данные графика
                data_file = f"training_data_{game_num}.npz"
                if os.path.exists(data_file):
                    data = np.load(data_file)
                    plot_scores = data["scores"].tolist()
                    plot_mean_scores = data["mean_scores"].tolist()
                    total_score = int(data["total_score"])
                    record = int(data["record"])
                    loaded_n_games = len(plot_scores)
                    agent.n_games = loaded_n_games

                    print(f"📊 Загружена модель: {model_to_load}")
                    print(f"📈 Загружены данные графика: {loaded_n_games} игр, рекорд: {record}")
                else:
                    print(f"⚠️ Модель найдена, но данные графика {data_file} не найдены")
                    # Если данных нет, начинаем график заново, но с моделью
                    print("📊 График начинается с нуля")
                    agent.n_games = game_num  # Устанавливаем номер игры из имени модели

    if not LOAD_MODEL:
        print("🚀 Режим 'с нуля': начинаем новое обучение")
        agent = Agent()

    env = SnakeEnv(render_mode="human")

    test_state = agent.get_state(env)
    print(f"✅ State shape: {test_state.shape}")
    print(f"✅ State range: [{test_state.min():.3f}, {test_state.max():.3f}]")
    # сообщение о типе сети
    print(f"🎯 Начинаем обучение с DUELING DDQN (Dueling Double Deep Q-Network)")
    print(f"🔧 Параметры: LR={LR}, Gamma={agent.gamma}, Tau={agent.trainer.tau}")
    print(f"🏗️  Архитектура: Dueling CNN (Value + Advantage streams)")

    while True:
        state_old = agent.get_state(env)
        final_move = agent.get_action(state_old)
        next_state, reward, done, ate, crashed = env.step(np.argmax(final_move))

        if agent.n_games % 10 == 0:
            print(f"Game {agent.n_games}, Action: {np.argmax(final_move)}, Reward: {reward:.2f}, "
                  f"Score: {env.score}, Epsilon: {agent.epsilon}")

        agent.train_short_memory(state_old, final_move, reward, next_state, done)
        agent.remember(state_old, final_move, reward, next_state, done)

        if done:
            score = env.score
            env.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                print(f"🎉 НОВЫЙ РЕКОРД! Score: {score}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            print(f'Game {agent.n_games}, Score {score}, Record: {record}, '
                  f'Epsilon: {agent.epsilon}, Learn steps: {agent.learn_step_counter}')

            # === Сохраняем каждые 5 игр ===
            if agent.n_games % 500 == 0:
                # Сохраняем модель
                model_name = f"model_dueling_{agent.n_games}.pth"
                torch.save(agent.model.state_dict(), model_name)

                # Сохраняем данные графика
                data_name = f"training_data_{agent.n_games}.npz"
                np.savez(data_name,
                         scores=plot_scores,
                         mean_scores=plot_mean_scores,
                         total_score=total_score,
                         record=record)

                print(f"💾 Сохранено: {model_name} и {data_name}")


if __name__ == '__main__':
    train()







# import os
# import torch
# import random
# import numpy as np
#
# from collections import deque
# from snake_env import SnakeEnv
# # === DUELING: Импортируем новую модель ===
# from model import DuelingCNN_QNet, QTrainer  # Изменен импорт!
# from helper import plot
#
# MAX_MEMORY = 32384
# BATCH_SIZE = 128
# LR = 0.0001
#
#
# class Agent:
#     def __init__(self):
#         self.n_games = 0
#         self.epsilon = 700
#         self.gamma = 0.9
#         self.memory = deque(maxlen=MAX_MEMORY)
#
#         # DDQN: Добавляем счетчик шагов для периодического обновления
#         self.learn_step_counter = 0
#         self.target_update_freq = 1000
#
#         # === DUELING: Меняем модель на DuelingCNN_QNet ===
#         self.model = DuelingCNN_QNet(input_channels=4, num_actions=4)  # Изменено!
#         # === DUELING: QTrainer теперь работает с Dueling сетью ===
#         self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, tau=0.005)
#
#     # метод наблюдения
#     def get_state(self, env):
#         return env._get_observation()
#
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
#
#     def train_long_memory(self):
#         if len(self.memory) > BATCH_SIZE:
#             mini_sample = random.sample(self.memory, BATCH_SIZE)
#         else:
#             mini_sample = self.memory
#
#         states, actions, rewards, next_states, dones = zip(*mini_sample)
#
#         states = np.array(states, dtype=np.float32)
#         next_states = np.array(next_states, dtype=np.float32)
#         actions = list(actions)
#         rewards = list(rewards)
#         dones = list(dones)
#
#         # Обучаем и получаем loss
#         loss = self.trainer.train_step(states, actions, rewards, next_states, dones)
#
#         self.learn_step_counter += 1
#
#         if self.learn_step_counter % self.target_update_freq == 0:
#             self.trainer.hard_update()
#             print(f"🔄 Таргетная сеть полностью обновлена (шаг {self.learn_step_counter})")
#             # === DUELING: Добавим информацию о типе сети ===
#             print(f"📊 Loss: {loss:.4f}, Games: {self.n_games}, Network: Dueling DDQN")
#
#         return loss
#
#     def train_short_memory(self, state, action, reward, next_state, done):
#         loss = self.trainer.train_step(
#             np.array([state], dtype=np.float32),
#             [action],
#             [reward],
#             np.array([next_state], dtype=np.float32),
#             [done]
#         )
#
#         self.learn_step_counter += 1
#
#         if self.learn_step_counter % self.target_update_freq == 0:
#             self.trainer.hard_update()
#             print(f"🔄 Таргетная сеть полностью обновлена (шаг {self.learn_step_counter})")
#
#         return loss
#
#     def get_action(self, state):
#         final_move = [0, 0, 0, 0]
#         # === DUELING: Включите epsilon-decay когда будете готовы ===
#         # self.epsilon = max(0, self.epsilon - 1)  # Пример decay
#         # self.epsilon = max(0, 700 - self.n_games)  # Линейный decay
#
#         # Exploration
#         if random.randint(0, 1300) < self.epsilon:
#             move = random.randint(0, 3)
#             final_move[move] = 1
#         else:
#             # === DUELING: Модель теперь Dueling, но интерфейс тот же ===
#             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
#
#             with torch.no_grad():
#                 prediction = self.model(state_tensor)
#
#             move = torch.argmax(prediction).item()
#             final_move[move] = 1
#
#         return final_move
#
#
# def train():
#     plot_scores = []
#     plot_mean_scores = []
#     total_score = 0
#     record = 0
#
#     agent = Agent()
#     env = SnakeEnv(render_mode="human")
#
#     test_state = agent.get_state(env)
#     print(f"✅ State shape: {test_state.shape}")
#     print(f"✅ State range: [{test_state.min():.3f}, {test_state.max():.3f}]")
#     # === DUELING: Обновляем сообщение о типе сети ===
#     print(f"🎯 Начинаем обучение с DUELING DDQN (Dueling Double Deep Q-Network)")
#     print(f"🔧 Параметры: LR={LR}, Gamma={agent.gamma}, Tau={agent.trainer.tau}")
#     print(f"🏗️  Архитектура: Dueling CNN (Value + Advantage streams)")
#
#     while True:
#         state_old = agent.get_state(env)
#         final_move = agent.get_action(state_old)
#         next_state, reward, done, ate, crashed = env.step(np.argmax(final_move))
#
#         if agent.n_games % 10 == 0:
#             print(f"Game {agent.n_games}, Action: {np.argmax(final_move)}, Reward: {reward:.2f}, "
#                   f"Score: {env.score}, Epsilon: {agent.epsilon}")
#
#         agent.train_short_memory(state_old, final_move, reward, next_state, done)
#         agent.remember(state_old, final_move, reward, next_state, done)
#
#         if done:
#             score = env.score
#             env.reset()
#             agent.n_games += 1
#             agent.train_long_memory()
#
#             if score > record:
#                 record = score
#                 print(f"🎉 НОВЫЙ РЕКОРД! Score: {score}")
#
#             # === DUELING: Обновляем имя файла для сохранения ===
#             if agent.n_games % 500 == 0:
#                 save_path = f"model_game_dueling_ddqn_{agent.n_games}_step_{agent.learn_step_counter}.pth"
#                 torch.save(agent.model.state_dict(), save_path)
#                 print(f"💾 Модель сохранена: {save_path} (Dueling DDQN)")
#
#             print(f'Game {agent.n_games}, Score {score}, Record: {record}, '
#                   f'Epsilon: {agent.epsilon}, Learn steps: {agent.learn_step_counter}')
#
#             plot_scores.append(score)
#             total_score += score
#             mean_score = total_score / agent.n_games
#             plot_mean_scores.append(mean_score)
#             plot(plot_scores, plot_mean_scores)
#
#
# if __name__ == '__main__':
#     train()