import torch
import numpy as np
from snake_env import SnakeEnv
from model import CNN_QNet
import pygame
from helper import plot


class SnakeTester:
    def __init__(self, model_path):
        # Инициализация среды с визуализацией
        self.env = SnakeEnv(render_mode="human")

        # Загрузка модели
        self.model = CNN_QNet(input_channels=4, num_actions=4)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        print(f"✅ Модель загружена из {model_path}")

        # Для графика
        self.test_scores = []
        self.mean_scores = []
        self.total_score = 0

    def get_state(self, env):
        """Получение состояния из среды"""
        return env._get_observation()

    def get_action(self, state):
        """Получение действия от модели (без случайности)"""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(state_tensor)
        move = torch.argmax(prediction).item()

        final_move = [0, 0, 0, 0]
        final_move[move] = 1
        return final_move

    def test(self, num_games=10, max_steps=1000):
        """Запуск тестирования"""
        print(f"\n🚀 Начинаю тестирование на {num_games} играх")

        for game in range(1, num_games + 1):
            state = self.env.reset()
            total_reward = 0
            done = False
            steps = 0

            print(f"\n🎮 Игра {game}/{num_games}")

            while not done and steps < max_steps:
                # Получаем действие
                action_onehot = self.get_action(state)
                action = np.argmax(action_onehot)

                # Выполняем шаг
                next_state, reward, done, ate, crashed = self.env.step(action)

                # Отслеживаем статистику
                total_reward += reward
                steps += 1

                # Обновляем состояние
                state = next_state

                # Обработка событий PyGame (чтобы окно не зависало)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return

                # Небольшая задержка для удобного просмотра
                # pygame.time.delay(50)

            # Сохраняем результаты игры для графика
            score = self.env.score
            self.test_scores.append(score)
            self.total_score += score
            mean_score = self.total_score / game
            self.mean_scores.append(mean_score)

            # Обновляем график с помощью существующего helper.plot()
            plot(self.test_scores, self.mean_scores)

            # Результаты игры
            print(f"   Результат: Счёт = {self.env.score}, Шагов = {steps}, Награда = {total_reward:.1f}")

            # Небольшая пауза между играми
            # pygame.time.delay(500)

        # Финальная статистика
        print(f"\n📊 Финальная статистика тестирования:")
        print(f"   Всего игр: {num_games}")
        print(f"   Средний счёт: {self.mean_scores[-1]:.2f}")
        print(f"   Максимальный счёт: {max(self.test_scores)}")
        print(f"   Минимальный счёт: {min(self.test_scores)}")

        print(f"\n✅ Тестирование завершено")
        pygame.time.delay(1000)
        pygame.quit()


def main():
    # Путь к сохранённой модели
    MODEL_PATH = "/Users/artemhorkov/desktop/RL/snake/model/model_game_20500.pth"

    # Создаём тестер
    tester = SnakeTester(MODEL_PATH)

    # Запускаем тестирование
    tester.test(num_games=50, max_steps=2000)


if __name__ == "__main__":
    main()
