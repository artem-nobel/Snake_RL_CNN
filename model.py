import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


# DUELING: новая архитектура
class DuelingCNN_QNet(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.num_actions = num_actions

        # DUELING: Общая часть
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # === DUELING: Общий feature layer вместо прямого fc ===
        self.feature_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # DUELING: Разделение на две ветки
        # Ветка VALUE: оценивает ценность состояние
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # ТОЛЬКО 1 ВЫХОД - V(s)
        )

        # Ветка ADVANTAGE: оценивает преимущество действия
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)  # num_actions выходов - A(s,a)
        )

    def forward(self, x):
        # DUELING: Прямой проход
        # 1. Извлекаем фичи
        x = self.conv(x)
        features = self.feature_layer(x)

        # 2. Вычисляем Value и Advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # DUELING: Объединение по формуле Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)


class QTrainer:
    # DUELING: конструктор
    def __init__(self, model, lr, gamma, tau=0.005):
        self.lr = lr
        self.gamma = gamma
        self.tau = tau  #DDQN Коэффициент обновления
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        # DUELING: Таргетная сеть
        self.target_model = DuelingCNN_QNet(model.conv[0].in_channels,
                                            model.num_actions)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # DDQN: Таргетная сеть оценкa

    # === DUELING: обновления
    def soft_update(self):
        # DDQN: мягкое обновление таргетной сети"""
        for target_param, local_param in zip(self.target_model.parameters(),
                                             self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data +
                                    (1.0 - self.tau) * target_param.data)

    def hard_update(self):
        # DDQN: Полное обновление таргетной сети
        self.target_model.load_state_dict(self.model.state_dict())

    # DUELING: Метод train_step
    def train_step(self, state, action, reward, next_state, done):
        # Конвертируем в тензоры
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Actions как тензор индексов
        actions_indices = []
        for act in action:
            if isinstance(act, list):
                actions_indices.append(torch.tensor(act, dtype=torch.long).argmax())
            else:
                actions_indices.append(act)

        action_tensor = torch.tensor(actions_indices, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        done_tensor = torch.tensor(done, dtype=torch.bool)

        # 1: Получаем предсказания для текущего состояния
        pred = self.model(state)

        # 2: Вычисляем target Q values с Double DQN логикой
        target = pred.clone()

        with torch.no_grad():
            # DDQN ШАГ 1: Используем основную сеть для выбора действий
            next_pred_main = self.model(next_state)
            next_actions = torch.argmax(next_pred_main, dim=1)

            # DDQN ШАГ 2: Используем таргетную сеть для оценки значений
            next_pred_target = self.target_model(next_state)
            max_next_q = next_pred_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # Обновляем target values
        for idx in range(len(done)):
            if done_tensor[idx]:
                Q_new = reward_tensor[idx]
            else:
                Q_new = reward_tensor[idx] + self.gamma * max_next_q[idx]

            target[idx][action_tensor[idx]] = Q_new

        # 3: Вычисляем loss и делаем backward pass
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        # 4: Обновляем веса
        self.optimizer.step()

        # DDQN: Мягкое обновление таргетной сети после каждого шага обучения
        self.soft_update()

        return loss.item()