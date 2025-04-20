# ### Project infant's Retionpathy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score

# Попробуем установить метод запуска процессов, но только если он еще не установлен
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass  # Метод уже установлен, пропускаем

# Параметры
DATA_DIR = '/Users/sergej/Downloads/archive/images_stack_without_captions/images_stack_without_captions'
BATCH_SIZE = 32
IMG_SIZE = 224
NUM_EPOCHS = 20
LR = 0.001
NUM_CLASSES = 2  # бинарная классификация (здоровые/больные)

# Создадим DataFrame с метаданными из имен файлов
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
train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(train_ids, test_size=0.25, random_state=42)  # 60/20/20 split

train_df = metadata_df[metadata_df['patient_id'].isin(train_ids)]
val_df = metadata_df[metadata_df['patient_id'].isin(val_ids)]
test_df = metadata_df[metadata_df['patient_id'].isin(test_ids)]

# Проверим баланс классов
print("Train class distribution:")
print(train_df['label'].value_counts())
print("\nValidation class distribution:")
print(val_df['label'].value_counts())
print("\nTest class distribution:")
print(test_df['label'].value_counts())

# Определим трансформации для изображений
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Создадим Dataset класс
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

# Создаем DataLoader'ы
train_dataset = RetinalDataset(train_df, DATA_DIR, transform=train_transform)
val_dataset = RetinalDataset(val_df, DATA_DIR, transform=val_test_transform)
test_dataset = RetinalDataset(test_df, DATA_DIR, transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


# Объявляем структуру модели
class RetinalModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(RetinalModel, self).__init__()
        # Используем предобученную ResNet18 в качестве базовой модели
        self.base_model = models.resnet18(pretrained=True)

        # Заменяем последний слой под нашу задачу
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

model = RetinalModel()
print(model)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')

    # Для ROC-AUC
    all_train_labels = []
    all_train_probs = []
    all_val_labels = []
    all_val_probs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Сохраняем для ROC-AUC
            probs = torch.softmax(outputs, dim=1)
            all_train_labels.extend(labels.cpu().numpy())
            all_train_probs.extend(probs[:, 1].cpu().detach().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Валидация
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Сохраняем для ROC-AUC
                probs = torch.softmax(outputs, dim=1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_probs.extend(probs[:, 1].cpu().detach().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Вычисляем ROC-AUC
        train_auc = roc_auc_score(all_train_labels, all_train_probs)
        val_auc = roc_auc_score(all_val_labels, all_val_probs)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print(f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
        print(f'Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
        print('-' * 50)

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    # Построение графиков
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    return train_losses, val_losses, train_accs, val_accs

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
    # Уменьшаем количество workers для DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Инициализация модели, функции потерь и оптимизатора
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinalModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR) # Добавлена L2 регуляризация после 1 теста

    # Обучение модели
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=NUM_EPOCHS
    )

    # Оценка на тестовом наборе
    print("\nFinal Evaluation on Test Set:")
    test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate_model(model, test_loader)

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
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

"""
Train class distribution:
label
0    1760
1    1343
Name: count, dtype: int64

Validation class distribution:
label
1    1186
0     724
Name: count, dtype: int64

Test class distribution:
label
0    496
1    495
Name: count, dtype: int64
/Users/sergej/Projects/.venv/lib/python3.13/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/Users/sergej/Projects/.venv/lib/python3.13/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
RetinalModel(
  (base_model): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer2): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer3): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (layer4): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): BasicBlock(
        (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Sequential(
      (0): Linear(in_features=512, out_features=512, bias=True)
      (1): ReLU()
      (2): Dropout(p=0.5, inplace=False)
      (3): Linear(in_features=512, out_features=2, bias=True)
    )
  )
)





Epoch 1/20
Train Loss: 0.6038, Val Loss: 0.7103
Train Acc: 0.6890, Val Acc: 0.5471
Train AUC: 0.7407, Val AUC: 0.8111
--------------------------------------------------
Epoch 2/20
Train Loss: 0.5431, Val Loss: 0.5607
Train Acc: 0.7396, Val Acc: 0.6963
Train AUC: 0.7719, Val AUC: 0.7892
--------------------------------------------------
Epoch 3/20
Train Loss: 0.4775, Val Loss: 0.7240
Train Acc: 0.7834, Val Acc: 0.6225
Train AUC: 0.8005, Val AUC: 0.7993
--------------------------------------------------
Epoch 4/20
Train Loss: 0.4627, Val Loss: 0.5050
Train Acc: 0.7973, Val Acc: 0.7592
Train AUC: 0.8159, Val AUC: 0.8086
--------------------------------------------------
Epoch 5/20
Train Loss: 0.4435, Val Loss: 0.5132
Train Acc: 0.8089, Val Acc: 0.7304
Train AUC: 0.8285, Val AUC: 0.8049
--------------------------------------------------
Epoch 6/20
Train Loss: 0.4238, Val Loss: 0.8825
Train Acc: 0.8150, Val Acc: 0.6120
Train AUC: 0.8393, Val AUC: 0.7872
--------------------------------------------------
Epoch 7/20
Train Loss: 0.4328, Val Loss: 0.6813
Train Acc: 0.8112, Val Acc: 0.7623
Train AUC: 0.8446, Val AUC: 0.7731
--------------------------------------------------
Epoch 8/20
Train Loss: 0.3870, Val Loss: 0.5241
Train Acc: 0.8356, Val Acc: 0.7346
Train AUC: 0.8531, Val AUC: 0.7806
--------------------------------------------------
Epoch 9/20
Train Loss: 0.3720, Val Loss: 0.6639
Train Acc: 0.8369, Val Acc: 0.6885
Train AUC: 0.8602, Val AUC: 0.7878
--------------------------------------------------
Epoch 10/20
Train Loss: 0.3446, Val Loss: 1.9480
Train Acc: 0.8543, Val Acc: 0.5822
Train AUC: 0.8674, Val AUC: 0.7764
--------------------------------------------------
Epoch 11/20
Train Loss: 0.3513, Val Loss: 0.5753
Train Acc: 0.8495, Val Acc: 0.7152
Train AUC: 0.8728, Val AUC: 0.7795
--------------------------------------------------
Epoch 12/20
Train Loss: 0.3313, Val Loss: 0.6115
Train Acc: 0.8704, Val Acc: 0.7073
Train AUC: 0.8784, Val AUC: 0.7816
--------------------------------------------------
Epoch 13/20
Train Loss: 0.3130, Val Loss: 0.8134
Train Acc: 0.8766, Val Acc: 0.6775
Train AUC: 0.8837, Val AUC: 0.7838
--------------------------------------------------
Epoch 14/20
Train Loss: 0.2961, Val Loss: 0.6269
Train Acc: 0.8798, Val Acc: 0.7079
Train AUC: 0.8888, Val AUC: 0.7876
--------------------------------------------------
Epoch 15/20
Train Loss: 0.2695, Val Loss: 0.6376
Train Acc: 0.8911, Val Acc: 0.7251
Train AUC: 0.8942, Val AUC: 0.7885
--------------------------------------------------
Epoch 16/20
Train Loss: 0.2674, Val Loss: 1.1702
Train Acc: 0.8869, Val Acc: 0.6492
Train AUC: 0.8989, Val AUC: 0.7888
--------------------------------------------------
Epoch 17/20
Train Loss: 0.2439, Val Loss: 0.5315
Train Acc: 0.8998, Val Acc: 0.7639
Train AUC: 0.9036, Val AUC: 0.7942
--------------------------------------------------
Epoch 18/20
Train Loss: 0.2271, Val Loss: 1.1203
Train Acc: 0.9098, Val Acc: 0.6728
Train AUC: 0.9082, Val AUC: 0.7967
--------------------------------------------------
Epoch 19/20
Train Loss: 0.2290, Val Loss: 0.6528
Train Acc: 0.9091, Val Acc: 0.7545
Train AUC: 0.9121, Val AUC: 0.7972
--------------------------------------------------
Epoch 20/20
Train Loss: 0.2129, Val Loss: 1.3376
Train Acc: 0.9201, Val Acc: 0.6634
Train AUC: 0.9161, Val AUC: 0.7952

Final Evaluation on Test Set:
Test Accuracy: 0.7730
Test Precision: 0.8068
Test Recall: 0.7172
Test F1-score: 0.7594
Test AUC: 0.8274

Classification Report:
Accuracy: 0.7730
Precision: 0.8068
Recall: 0.7172
F1-score: 0.7594
AUC-ROC: 0.8274

Итог 
1) Проблемы с переобучением с 10 эпохи
2)не хватает L2 регуляризации
3) нужна аугментация данных
4) нестабильный и низкий val loss


Final Evaluation on Test Set:
Test Accuracy: 0.7830
Test Precision: 0.8241
Test Recall: 0.7192
Test F1-score: 0.7681
Test AUC: 0.8414

Classification Report:
Accuracy: 0.7830
Precision: 0.8241
Recall: 0.7192
F1-score: 0.7681
AUC-ROC: 0.8414




Распределение тренировочной выборки:
label
0    1760
1    1343
Name: count, dtype: int64

Распределение валидационной выборки:
label
1    1186
0     724
Name: count, dtype: int64

Распределение тестовой выборки:
label
0    496
1    495
"""
