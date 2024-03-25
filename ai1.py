import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Определение архитектуры нейронной сети
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3 * 3, 16)  # Входной слой: 3x3 пикселя, 16 скрытых нейронов
        self.fc2 = nn.Linear(16, 1)     # Выходной слой: 1 выход (есть точка или нет)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Преобразование входных данных в одномерный тензор
        x = torch.sigmoid(self.fc1(x))     # Применение сигмоидной функции активации к скрытому слою
        x = torch.sigmoid(self.fc2(x))     # Применение сигмоидной функции активации к выходному слою
        return x

# Создание экземпляра нейронной сети
net = SimpleNet()

# Определение функции потерь и оптимизатора
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Обучение нейронной сети
num_epochs = 100

# Генерация обучающих данных
train_data = np.random.randint(0, 256, size=(100, 3, 3))  # Пример случайной генерации данных
train_labels = np.random.randint(0, 2, size=(100, 1))     # Пример случайной генерации меток

for epoch in range(num_epochs):
    inputs = torch.FloatTensor(train_data)
    labels = torch.FloatTensor(train_labels)

    # Обнуление градиентов
    optimizer.zero_grad()

    # Прямой проход (подсчет предсказаний)
    outputs = net(inputs)

    # Вычисление функции потерь
    loss = criterion(outputs, labels)

    # Обратный проход (вычисление градиентов)
    loss.backward()

    # Обновление весов
    optimizer.step()

    # Вывод промежуточной информации
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Тестирование обученной нейронной сети
test_data = np.random.randint(0, 256, size=(10, 3, 3))  # Пример случайной генерации тестовых данных

with torch.no_grad():
    inputs = torch.FloatTensor(test_data)
    outputs = net(inputs)
    predictions = (outputs >= 0.5).int().numpy()

    print("Test Predictions:")
    for i, prediction in enumerate(predictions):
        print(f"Image {i+1}: {'Point exists' if prediction[0] == 1 else 'No point'}")