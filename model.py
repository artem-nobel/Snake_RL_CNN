import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class CNN_QNet(nn.Module):
    def __init__(self, input_channels, num_actions):
        super().__init__()
        # свертки 3x3, stride=1, padding=1
        self.conv = nn.Sequential(
            # Первый слой: из 4 каналов в 32
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=1),
            nn.ReLU(),

            # Второй слой: из 32 в 64
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # nn.Dropout2d(0.05),
            # Третий слой: из 64 в 64
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Dropout2d(0.05)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_actions)
        )


    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

    def save(self, file_name):
        torch.save(self.model.state_dict(), file_name)

    # def load(self, file_name):
    #     if os.path.exists(file_name):
    #         self.load_state_dict(torch.load(file_name, map_location=torch.device('cpu')))
    #         self.eval()  # переводим в режим оценки
    #         print(f"✅ Модель загружена из {file_name}")
    #         return True
    #     else:
    #         print(f"⚠️ Файл {file_name} не найден, начинаем с нуля")
    #         return False

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):

        # state уже должен быть numpy array с shape (batch, 4, 84, 84)

        # Конвертируем в тензоры
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)

        # Actions как тензор индексов
        actions_indices = []
        for act in action:
            # Находим индекс выбранного действия (где 1 в one-hot векторе)
            if isinstance(act, list):
                actions_indices.append(torch.tensor(act, dtype=torch.long).argmax())
            else:
                actions_indices.append(act)  # если уже индекс

        action_tensor = torch.tensor(actions_indices, dtype=torch.long)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        done_tensor = torch.tensor(done, dtype=torch.bool)

        # 1: Получаем предсказания для текущего состояния
        pred = self.model(state)

        # 2: Вычисляем target Q values
        target = pred.clone()

        with torch.no_grad():
            next_pred = self.model(next_state)
            max_next_q = torch.max(next_pred, dim=1)[0]

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

        return loss.item()

