# Третья версия 
# Используем ResNet50 вместо ResNet18
# Улучшенная аугментация данных 
# Оптимизация процесса обучения

import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score
import numpy as np

# Параметры
DATA_DIR = '/Users/sergej/Downloads/archive/images_stack_without_captions/images_stack_without_captions'
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS = 30  # Увеличено количество эпох
LR = 0.001
NUM_CLASSES = 2

# Установка метода запуска процессов
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass


def create_metadata_df(data_dir):
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    data = []

    for img_file in image_files:
        parts = img_file.split('_')
        if len(parts) >= 9:  # Убедимся, что имя файла содержит все необходимые части
            patient_id = parts[0]
            sex = parts[1]
            ga = int(parts[2][2:])
            bw = int(parts[3][2:])
            pa = int(parts[4][2:])
            dg = int(parts[5][2:])
            pf = int(parts[6][2:])
            device = parts[7][1:]
            series = parts[8][1:]
            img_num = parts[9].split('.')[0]

            # Бинарная метка: 0 - здоровый, 1 - больной (на основе диагноза)
            label = 0 if dg == 0 else 1  # Предполагаем, что DG=0 означает здоровый

            data.append({
                'filename': img_file,
                'patient_id': patient_id,
                'sex': sex,
                'ga': ga,
                'bw': bw,
                'pa': pa,
                'dg': dg,
                'pf': pf,
                'device': device,
                'series': series,
                'img_num': img_num,
                'label': label
            })

    return pd.DataFrame(data)


# Создаем DataFrame с метаданными
metadata_df = create_metadata_df(DATA_DIR)

# print(metadata_df.head())

# Разделение на train/val/test с учетом patient_id (чтобы изображения одного пациента не попали в разные наборы)
patient_ids = metadata_df['patient_id'].unique()
train_ids, test_ids = train_test_split(
    patient_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(
    train_ids, test_size=0.25, random_state=42)  # 60/20/20 split

train_df = metadata_df[metadata_df['patient_id'].isin(train_ids)]
val_df = metadata_df[metadata_df['patient_id'].isin(val_ids)]
test_df = metadata_df[metadata_df['patient_id'].isin(test_ids)]


# Улучшенная аугментация данных
# train_transform = transforms.Compose([
#     transforms.Resize(256),                          # Изменение размера до 256x256
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Случайный кроп 224x224
#     transforms.RandomHorizontalFlip(),               # Горизонтальное отражение (вероятность 50%)
#     transforms.RandomVerticalFlip(),                 # Вертикальное отражение (вероятность 50%)
#     transforms.RandomRotation(20),                   # Поворот на ±20 градусов
#     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),  # Изменение цвета
#     transforms.ToTensor(),                           # Конвертация в тензор
#     transforms.Normalize(                             # Нормализация
#         mean=[0.485, 0.456, 0.406],                 # Средние значения ImageNet
#         std=[0.229, 0.224, 0.225]                   # Стандартные отклонения ImageNet
#     ),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))  # Случайное удаление областей
# ])

train_transform = transforms.Compose([
    transforms.Resize(280),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.15), value='random')
])


val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),        # Фиксированный размер 224x224
    transforms.ToTensor(),                           # Конвертация в тензор
    transforms.Normalize(                            # Нормализация (как для train)
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])




class EarlyStopping:
    def __init__(self, patience=7, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False




class RetinalDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label


class RetinalModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(RetinalModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True) 
        in_features = self.base_model.fc.in_features
        
        # Улучшенная головная часть
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epochs=NUM_EPOCHS):
    train_losses = []
    val_losses = []
    train_recalls = []
    val_recalls = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_recall = recall_score(all_train_labels, all_train_preds)

        # Валидация
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_recall = recall_score(all_val_labels, all_val_preds)

        # Обновление scheduler и early stopping
        scheduler.step(val_loss)
        if early_stopping(val_loss):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

        # Сохранение метрик
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(
            f'Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f}')
        print('-' * 60)

    # Визуализация
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(train_recalls, label='Train Recall')
    plt.plot(val_recalls, label='Val Recall')
    plt.legend()
    plt.title('Recall Curves')
    plt.show()

    return model


def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f'Test Accuracy: {accuracy:.4f}')
    print(f'Test Precision: {precision:.4f}')
    print(f'Test Recall: {recall:.4f}')
    print(f'Test F1-score: {f1:.4f}')
    print(f'Test AUC: {auc:.4f}')

    return accuracy, precision, recall, f1, auc


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metadata_df = create_metadata_df(DATA_DIR)

    train_dataset = RetinalDataset(train_df, DATA_DIR, train_transform)
    val_dataset = RetinalDataset(val_df, DATA_DIR, val_test_transform)
    test_dataset = RetinalDataset(test_df, DATA_DIR, val_test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Инициализация модели
    model = RetinalModel().to(device)

    # # Взвешенная функция потерь для дисбаланса классов
    # class_weights = torch.tensor([1.0, 3.0]).to(
    #     device)  # Больший вес для класса ROP
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # # Оптимизатор с L2-регуляризацией
    # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # # Scheduler и Early Stopping
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min',       # Следим за снижением val_loss
    #     factor=0.5,      # Умножаем lr на 0.5 при срабатывании
    #     patience=3,      # Ждём 3 эпохи без улучшений
    #     verbose=True     # Вывод сообщений в консоль
    # )

    # Функция потерь с более сбалансированными весами
    class_weights = torch.tensor([1.0, 2.0]).to(device)  # Было [1.0, 3.0]
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Оптимизатор с измененными параметрами
    optimizer = optim.AdamW(model.parameters(), 
                        lr=0.0005,  # Уменьшен learning rate
                        weight_decay=1e-5)  # Уменьшен weight decay

    # Измененный scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=10,  # Период сброса lr
        eta_min=1e-6  # Минимальный lr
    )


    early_stopping = EarlyStopping(patience=7, delta=0.001)

    # Вывод модели и распределения
    print(model)
    # Проверим баланс классов
    print("Train class distribution:")
    print(train_df['label'].value_counts())
    print("\nValidation class distribution:")
    print(val_df['label'].value_counts())
    print("\nTest class distribution:")
    print(test_df['label'].value_counts())

    # Обучение
    model = train_model(model, train_loader, val_loader,
                        criterion, optimizer, scheduler, early_stopping)

    # Оценка
    print("\nFinal Evaluation on Test Set:")
    test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate_model(
        model, test_loader)

    # Детальный отчет
    # Выведем основные метрики
    print("\nClassification Report:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")

    # Построение ROC-кривой
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

""""
Epoch 1/30
Train Loss: 0.6063 | Val Loss: 0.6680
Train Recall: 0.7692 | Val Recall: 0.5860
------------------------------------------------------------

Epoch 2/30
Train Loss: 0.5504 | Val Loss: 0.7020
Train Recall: 0.7878 | Val Recall: 0.4503
------------------------------------------------------------

Epoch 3/30
Train Loss: 0.5102 | Val Loss: 0.5526
Train Recall: 0.8012 | Val Recall: 0.6889
------------------------------------------------------------

Epoch 4/30
Train Loss: 0.5082 | Val Loss: 0.5248
Train Recall: 0.7870 | Val Recall: 0.7479
------------------------------------------------------------

Epoch 5/30
Train Loss: 0.5070 | Val Loss: 0.4289
Train Recall: 0.8101 | Val Recall: 0.8322
------------------------------------------------------------

Epoch 6/30
Train Loss: 0.5225 | Val Loss: 0.6118
Train Recall: 0.8064 | Val Recall: 0.8499
------------------------------------------------------------

Epoch 7/30
Train Loss: 0.4873 | Val Loss: 0.8026
Train Recall: 0.7997 | Val Recall: 0.6678
------------------------------------------------------------

Epoch 8/30
Train Loss: 0.4716 | Val Loss: 0.4404
Train Recall: 0.8057 | Val Recall: 0.7218
------------------------------------------------------------

Epoch 9/30
Train Loss: 0.4718 | Val Loss: 0.4212
Train Recall: 0.8094 | Val Recall: 0.7993
------------------------------------------------------------

Epoch 10/30
Train Loss: 0.4554 | Val Loss: 0.5086
Train Recall: 0.8421 | Val Recall: 0.7589
------------------------------------------------------------

Epoch 11/30
Train Loss: 0.4523 | Val Loss: 0.4466
Train Recall: 0.8183 | Val Recall: 0.8035
------------------------------------------------------------

Epoch 12/30
Train Loss: 0.4264 | Val Loss: 0.3533
Train Recall: 0.8414 | Val Recall: 0.8862
------------------------------------------------------------

Epoch 13/30
Train Loss: 0.4316 | Val Loss: 0.4343
Train Recall: 0.8474 | Val Recall: 0.7782
------------------------------------------------------------

Epoch 14/30
Train Loss: 0.4392 | Val Loss: 0.4622
Train Recall: 0.8265 | Val Recall: 0.7673
------------------------------------------------------------

Epoch 15/30
Train Loss: 0.4028 | Val Loss: 0.3628
Train Recall: 0.8690 | Val Recall: 0.8651
------------------------------------------------------------

Epoch 16/30
Train Loss: 0.4149 | Val Loss: 0.4446
Train Recall: 0.8317 | Val Recall: 0.7639
------------------------------------------------------------

Epoch 17/30
Train Loss: 0.4005 | Val Loss: 0.3726
Train Recall: 0.8518 | Val Recall: 0.8474
------------------------------------------------------------

Epoch 18/30
Train Loss: 0.3820 | Val Loss: 0.4885
Train Recall: 0.8585 | Val Recall: 0.9191
------------------------------------------------------------

Early stopping triggered at epoch 19

Final Evaluation on Test Set:
Test Accuracy: 0.6731
Test Precision: 0.6230
Test Recall: 0.8947
Test F1-score: 0.7277
Test AUC: 0.8413

Classification Report:
Accuracy: 0.6731
Precision: 0.6230
Recall: 0.8947
F1-score: 0.7277
AUC-ROC: 0.8413


"""