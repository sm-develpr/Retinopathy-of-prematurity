


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
NUM_EPOCHS = 30 
LR = 0.001
NUM_CLASSES = 2



#! Преобразует данные в начальный Dataframe с метаданными 
def create_df(data_dir):
    image_files = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    data = []

    for img_file in image_files:
        parts = img_file.split('_')
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
        # Бинарная метка: 0 - здоровый, 1 - больной
        label = 0 if dg == 0 else 1  
        data.append({
                'filename': img_file,
                'patient_id': patient_id,
                'sex': sex,
                'GESTATIONAL_AGE': ga,
                'BIRTH_WEIGHT': bw,
                'POSTCONCEPTUAL_AGE': pa,
                'DIAGNOSIS_CODE': dg,
                'PLUS_FORM': pf,
                'DEVICE': device,
                'SERIES_NUMBER': series,
                'img_num': img_num,
                'label': label
            })

    return pd.DataFrame(data)


# Создаем DataFrame с метаданными
metadata_df = create_df(DATA_DIR)

# Разделение на train/val/test с учетом patient_id (чтобы изображения одного пациента не попали в разные наборы)
patient_ids = metadata_df['patient_id'].unique()
train_ids, test_ids = train_test_split(
    patient_ids, test_size=0.2, random_state=42)
train_ids, val_ids = train_test_split(
    train_ids, test_size=0.25, random_state=42)  # 60/20/20 split

train_df = metadata_df[metadata_df['patient_id'].isin(train_ids)]
val_df = metadata_df[metadata_df['patient_id'].isin(val_ids)]
test_df = metadata_df[metadata_df['patient_id'].isin(test_ids)]

#! Создание трансформаций изоьражений и их нормализация

train_transform = transforms.Compose([
    transforms.Resize(280), # Изменение размера до 280x280

    # Случайный поворот (-15°,+15°), сдвиг до 10%, масштаб (90%-110%)
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    
    # Вырезает случайную область (70%-100% изображения) и масштабирует до 224x224
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), 
    transforms.RandomHorizontalFlip(), # Зеркальное отражение по горизонтали (50% шанс)
    transforms.RandomVerticalFlip(), # Зеркальное отражение по вертикали (50% шанс)

    # Случайные изменения: яркость, контраст, насыщенность (±20%), оттенок (±0.02)
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),

    # Размытие Гаусса (ядро 3x3, сила размытия 0.1-2.0)
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # Случайное затирание областей (50% шанс, размер 2%-15%, заполнение случайным цветом)
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



#! Функция  останавливает обучение, когда модель начинает "запоминать" тренировочные данные вместо обучения общим закономерностям
#! Избегает бесполезных вычислений после достижения оптимального результата.
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



#! Обрабатывает изображени
class RetinalDataset(Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        # Нормализуем числовые признаки
        self.numeric_cols = ['GESTATIONAL_AGE', 'BIRTH_WEIGHT', 'POSTCONCEPTUAL_AGE']
        self.means = self.df[self.numeric_cols].mean()
        self.stds = self.df[self.numeric_cols].std()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['label']
        
        # Получаем дополнительные признаки
        numeric_features = self.df.iloc[idx][self.numeric_cols].values.astype(np.float32)
        # Cтандартизируем, ну практически тоже самое что и Нормализация
        numeric_features = (numeric_features - self.means.values) / self.stds.values
        
        # Сделали бинарным пол 
        #? Может сделать по one-hot-encoding?
        sex = 1 if self.df.iloc[idx]['sex'] == 'M' else 0
        
        if self.transform:
            image = self.transform(image)
            
        # Возвращаем изображение, числовые признаки, категориальный признак и метку
        return image, torch.FloatTensor(numeric_features), torch.tensor(sex, dtype=torch.float32), label
    

#! Модель
class RetinalModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(RetinalModel, self).__init__()
        # Замораживаем часть слоев ResNet
        self.cnn = models.resnet50(pretrained=True)
        for param in list(self.cnn.parameters())[:-50]:  # Замораживаем все кроме последних параметров
            param.requires_grad = False
        self.cnn.fc = nn.Identity()
        
        # Упрощенный обработчик признаков с L2-регуляризацией
        self.feature_processor = nn.Sequential(
            nn.Linear(4, 32),
            nn.Dropout(0.4),
            nn.Linear(32, 64))
        
        # Более регуляризованный классификатор
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 256),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes))
        
    def forward(self, img, numeric_features, sex):
        img_features = self.cnn(img)
        additional_features = torch.cat([numeric_features, sex.unsqueeze(1)], dim=1)
        processed_features = self.feature_processor(additional_features)
        combined = torch.cat([img_features, processed_features], dim=1)
        return self.classifier(combined)


#! Функция для тренировки модели 
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

        for images, numeric, sex, labels in train_loader:
            images = images.to(device)
            numeric = numeric.to(device)
            sex = sex.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, numeric, sex)
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
            for images, numeric, sex, labels in val_loader:
                images = images.to(device)
                numeric = numeric.to(device)
                sex = sex.to(device)
                labels = labels.to(device)
                
                outputs = model(images, numeric, sex)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        val_loss /= len(val_loader)
        val_recall = recall_score(all_val_labels, all_val_preds)

        scheduler.step(val_loss)
        if early_stopping(val_loss):
            print(f'Остановлен после {epoch} эпохи')
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)

        print(f'Эпоха {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        print(f'Train Recall: {train_recall:.4f} | Val Recall: {val_recall:.4f}')
        print('-' * 60)

    # Визуализация
    plt.figure(figsize=(15, 5))

    # Функция потерь 
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Функция потерь')

    # Recall
    plt.subplot(1, 2, 2)
    plt.plot(train_recalls, label='Train Recall')
    plt.plot(val_recalls, label='Val Recall')
    plt.legend()
    plt.title('Кривая Recall')
    plt.show()

    return model

    
#! Функция для оценки модели 
def evaluate_model(model, test_loader, threshold=0.3):  # Добавляем параметр threshold
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, numeric, sex, labels in test_loader:
            images = images.to(device)
            numeric = numeric.to(device)
            sex = sex.to(device)
            labels = labels.to(device)
            
            outputs = model(images, numeric, sex)
            probs = torch.softmax(outputs, dim=1)
            
            # Изменяем способ получения предсказаний с учетом порога
            preds = (probs[:, 1] >= threshold).long()  # Используем заданный порог
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return accuracy, precision, recall, f1, auc


if __name__ == '__main__':
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    metadata_df = create_df(DATA_DIR)

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

   # Функция потерь с более сбалансированными весами
    class_weights = torch.tensor([1.0, 3.0]).to(device)  # Было [1.0, 2.0]
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Оптимизатор с измененными параметрами
    optimizer = optim.AdamW([
        {'params': [p for p in model.cnn.parameters() if p.requires_grad], 'lr': 1e-5},
        {'params': model.feature_processor.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-4)  # Увеличена регуляризация
    
    # Измененный scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, 
    #     T_max=10,  # Период сброса lr в эпохах
    #     eta_min=1e-6  # Минимальный lr
    # )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', # Следить за уменьшением val_loss
        factor=0.5, # Умножать LR на 0.5 при срабатывании
        patience=3, # Ждать 3 эпохи без улучшения
        verbose=True # Выводить сообщения
        )
    

    early_stopping = EarlyStopping(patience=5, delta=0.001) 


    # Вывод модели и распределения
    print(model)
    # Проверим баланс классов
    print("Распределение в обучающей выборке")
    print(train_df['label'].value_counts())
    print("\nРаспределение в валидационной выборке")
    print(val_df['label'].value_counts())
    print("\nРаспределение в тестовой выборке")
    print(test_df['label'].value_counts())

    # Обучение
    model = train_model(model, train_loader, val_loader,
                        criterion, optimizer, scheduler, early_stopping)

    # Оценка
    # Оценка с порогом 0.3
    print("\nФинальная оценка по тестовой выборке (порог 30%)")
    test_accuracy, test_precision, test_recall, test_f1, test_auc = evaluate_model(
        model, test_loader, threshold=0.3)  # Указываем порог 0.3

    # Детальный отчет
    print("\nClassification Report (порог 30%):")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1-score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")

    # Оценка с порогом 0.5 для сравнения
    print("\nДля сравнения - оценка с порогом 50%:")
    test_accuracy_50, test_precision_50, test_recall_50, test_f1_50, _ = evaluate_model(
        model, test_loader, threshold=0.5)

    print("\nClassification Report (порог 50%):")
    print(f"Accuracy: {test_accuracy_50:.4f}")
    print(f"Precision: {test_precision_50:.4f}")
    print(f"Recall: {test_recall_50:.4f}")
    print(f"F1-score: {test_f1_50:.4f}")
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, numeric, sex, labels in test_loader:  
            images = images.to(device)
            numeric = numeric.to(device)
            sex = sex.to(device)
            outputs = model(images, numeric, sex)  # Передаем все необходимые аргументы
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (area = {test_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
""""

RetinalModel(
  (cnn): ResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer2): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer3): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (3): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (4): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (5): Bottleneck(
        (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (layer4): Sequential(
      (0): Bottleneck(
        (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (downsample): Sequential(
          (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
      (2): Bottleneck(
        (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
      )
    )
    (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
    (fc): Identity()
  )
  (feature_processor): Sequential(
    (0): Linear(in_features=4, out_features=32, bias=True)
    (1): Dropout(p=0.4, inplace=False)
    (2): Linear(in_features=32, out_features=64, bias=True)
  )
  (classifier): Sequential(
    (0): Linear(in_features=2112, out_features=256, bias=True)
    (1): Dropout(p=0.6, inplace=False)
    (2): Linear(in_features=256, out_features=2, bias=True)
  )
)
Распределение в обучающей выборке
label
0    1760
1    1343
Name: count, dtype: int64

Распределение в валидационной выборке
label
1    1186
0     724
Name: count, dtype: int64

Распределение в тестовой выборке
label
0    496
1    495
Name: count, dtype: int64
Эпоха 1/30
Train Loss: 0.5623 | Val Loss: 0.3924
Train Recall: 0.9077 | Val Recall: 0.8769
------------------------------------------------------------
Эпоха 2/30
Train Loss: 0.4721 | Val Loss: 0.4361
Train Recall: 0.8794 | Val Recall: 0.8078
------------------------------------------------------------
Эпоха 3/30
Train Loss: 0.4218 | Val Loss: 0.3733
Train Recall: 0.8734 | Val Recall: 0.8398
------------------------------------------------------------
Эпоха 4/30
Train Loss: 0.3893 | Val Loss: 0.3255
Train Recall: 0.8943 | Val Recall: 0.8777
------------------------------------------------------------
Эпоха 5/30
Train Loss: 0.3749 | Val Loss: 0.3376
Train Recall: 0.8876 | Val Recall: 0.8735
------------------------------------------------------------
Эпоха 6/30
Train Loss: 0.3518 | Val Loss: 0.4708
Train Recall: 0.8950 | Val Recall: 0.7470
------------------------------------------------------------
Эпоха 7/30
Train Loss: 0.3262 | Val Loss: 0.4013
Train Recall: 0.9047 | Val Recall: 0.8153
------------------------------------------------------------
Эпоха 8/30
Train Loss: 0.3363 | Val Loss: 0.4825
Train Recall: 0.8920 | Val Recall: 0.7504
------------------------------------------------------------
Остановлен после 8 эпохи
2025-04-21 01:33:01.842 Python[69944:2928786] +[IMKClient subclass]: chose IMKClient_Legacy
2025-04-21 01:33:01.842 Python[69944:2928786] +[IMKInputSession subclass]: chose IMKInputSession_Legacy

Финальная оценка по тестовой выборке (порог 30%)

Classification Report (порог 30%):
Accuracy: 0.7326
Precision: 0.6753
Recall: 0.8949
F1-score: 0.7698
AUC-ROC: 0.8447

Для сравнения - оценка с порогом 50%:

Classification Report (порог 50%):
Accuracy: 0.7750
Precision: 0.7305
Recall: 0.8707
F1-score: 0.7945
"""