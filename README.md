# Development of a Model for Extracting Additional Visual Characteristics from Complex Scene Images

This project is dedicated to developing a model for converting three-channel images to five-channel and back. The main goal is to improve object recognition accuracy through image preprocessing. The detection model cannot process 5 channels, so the number of channels is limited to avoid uninterpretable results.

## Project Description

In the process of image conversion, it was decided to add two additional channels: depth (D) and contour (E). Thus, a five-channel image is formed: R, G, B, D, and E.

For converting five-channel images back to three-channel, the U-NET architecture was chosen. Despite high memory costs and limited flexibility, this architecture provides good results in image segmentation and restoration tasks.

### Model Improvements

To address the shortcomings of U-NET, the following improvements were proposed:

- **Channel Attention Mechanism**: This mechanism allows compressing information and creating weight coefficients for more accurate extraction of important features in images.

- **Depthwise and Pointwise Convolutions**: The use of these types of convolutions helps optimize resources and improve image restoration quality.

### Final Architecture

The final model architecture combines an encoder with a triple decoder with attention channels between them. This architecture allows channel-wise preservation of object features, systematically highlighting aspects of the data important for object representation.

## Technologies

- **Programming Languages:** Python
- **Deep Learning Framework:** PyTorch, PyTorch Lightning
- **Model Architecture:** U-NET with Channel Attention
- **Image Processing:** OpenCV, PIL, scikit-image
- **Data Manipulation:** NumPy, pandas
- **Visualization:** Matplotlib
- **Logging:** WandB
- **Object Detection:** YOLOv8 (Ultralytics)
- **Depth Estimation:** Depth-Anything, DepthPro (HuggingFace Transformers)
- **Metrics:** SSIM, PSNR, FSIM, mAP

## Workflow

1. **Data Loading:**
   - Load image datasets with bounding box annotations
   - Convert class labels to YOLO-compatible format
   - Split data into training and validation sets

2. **Five-Channel Image Generation:**
   - **RGB Channels:** Original color channels
   - **Depth Channel:** Generated using pre-trained depth estimation models (Depth-Anything or DepthPro)
   - **Contour Channel:** Generated using Canny edge detection

3. **Model Architecture:**
   - **Encoder:** Multi-level convolutional encoder with channel attention
   - **Decoder:** Triple-stream decoder (color, depth, contour branches)
   - **Skip Connections:** Feature propagation from encoder to decoder

4. **Loss Functions:**
   - **YOLOv8 Loss:** Combined loss for detection (box, cls, dfl)
   - **FOX Loss:** Multi-component loss for channel reconstruction
   - **RGBTargetAwareLoss:** Individual weights for color, depth, and contour
   - **Complex Loss:** Weighted combination of detection and reconstruction losses
   - **SSIM Loss:** Structural similarity loss with MSE component

5. **Metrics:**
   - **PSNR:** Peak Signal-to-Noise Ratio
   - **FSIM:** Feature Similarity Index
   - **mAP:** Mean Average Precision for object detection
   - **SSIM:** Structural Similarity Index

## Installation

### 1. Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### 2. Clone the Repository

```bash
git clone https://github.com/your_username/five-channel-image-processing.git
cd five-channel-image-processing
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

- YOLOv8x-oiv7.pt (pre-trained detection model)
- Depth-Anything-V2-Small-hf (depth estimation)
- DepthPro-hf (alternative depth estimation)

### 5. Run Training

```bash
python Main.py
```

## Model Architecture Details

### U-NET with Channel Attention

```
Input (3-channel RGB)
    ↓
Encoder Level 1: Conv2D(16) + MaxPool + ChannelAttention
    ↓
Encoder Level 2: Conv2D(32) + MaxPool + ChannelAttention
    ↓
Encoder Level 3: Conv2D(64) + MaxPool + ChannelAttention
    ↓
Encoder Level 4: Conv2D(128) + MaxPool + ChannelAttention
    ↓
Decoder Level 1: UpConv + Skip Connection (Color Branch)
    ↓
Decoder Level 2: UpConv + Skip Connection (Depth Branch)
    ↓
Decoder Level 3: UpConv + Skip Connection (Contour Branch)
    ↓
Output (3-channel RGB reconstruction)
```

## Results

### Visual Results

**a) CDE 1 DA** | **b) CDE 2 DA** | **c) CDE 3 DA** | **d) CDE 3 DP** | **e) Original**

![CDE 1 DA](https://github.com/user-attachments/assets/5e8edcc4-111c-4041-aa73-f5c32f84f6ef) | ![CDE 2 DA](https://github.com/user-attachments/assets/12b5ceab-a81c-4a47-bb8f-aab7f39d1e47) | ![CDE 3 DA](https://github.com/user-attachments/assets/5444a6a5-6d3a-4730-9335-061c433ced3d) | ![CDE 3 DP](https://github.com/user-attachments/assets/d7e0ae22-919a-46cd-8111-4006c3b04cbb) | ![Original](https://github.com/user-attachments/assets/28b9136a-c80c-4d78-8cb4-4df8fe852bdd)

*Figure 1: Examples of five-channel to three-channel reconstruction. a) CDE 1 DA, b) CDE 2 DA, c) CDE 3 DA, d) CDE 3 DP, e) Original image.*

### Accuracy Results

![Accuracy Results](https://github.com/user-attachments/assets/2f2fa563-7cd9-4a39-abbf-973ff39f0b06)

*Figure 2: Accuracy metrics for the model on new images.*

## Output

The system generates the following outputs:

1. **Reconstructed Images:** Three-channel images restored from five-channel input
2. **Detection Results:** Object detection with bounding boxes and class labels
3. **Training Logs:** Loss curves, gradient histograms, and metrics via WandB
4. **Model Weights:** Saved model checkpoints for inference
5. **Metrics:** PSNR, SSIM, FSIM, and mAP values

## Future Development

- Integration of additional depth estimation models
- Implementation of real-time inference pipeline
- Optimization for edge devices
- Multi-scale training strategy
- Additional data augmentation techniques
- Extension to video processing
- Implementation of model distillation for faster inference

---------------------------------------------------------

# Разработка модели выделения дополнительных визуальных характеристик изображения со сложной сценой

Данный проект посвящен разработке модели для преобразования трехканальных изображений в пятиканальные и обратно. Основная цель заключается в улучшении точности распознавания объектов с использованием предобработки изображений. Модель детектирования не может обрабатывать 5 каналов, поэтому выбор количества каналов ограничен, чтобы избежать неинтерпретируемых результатов.

## Описание проекта

В процессе преобразования изображений было решено добавить два дополнительных канала: глубинный (D) и контурный (E). Таким образом, формируется изображение из пяти каналов: R, G, B, D и E.

Для преобразования пятиканальных изображений обратно в трехканальные была выбрана архитектура U-NET. Несмотря на высокие затраты памяти и ограниченную гибкость данной архитектуры, она обеспечивает хорошие результаты в задачах сегментации и восстановления изображений.

### Улучшения модели

Для преодоления недостатков U-NET были предложены следующие улучшения:

- **Механизм Channel Attention**: Этот механизм позволяет сжимать информацию и создавать весовые коэффициенты для более точного выделения важных признаков в изображениях.

- **Depthwise и Pointwise свертки**: Использование этих типов сверток помогает оптимизировать ресурсы и улучшить качество восстановления изображений.

### Конечная архитектура

Конечная архитектура модели представляет собой сочетание энкодера и тройного декодера с каналами внимания между ними. Эта архитектура позволяет поканально сохранять особенности объектов изображения, систематически выделяя важные для представления объектов аспекты данных.

## Стек технологий

- **Языки программирования:** Python
- **Фреймворк глубокого обучения:** PyTorch, PyTorch Lightning
- **Архитектура модели:** U-NET с Channel Attention
- **Обработка изображений:** OpenCV, PIL, scikit-image
- **Обработка данных:** NumPy, pandas
- **Визуализация:** Matplotlib
- **Логирование:** WandB
- **Обнаружение объектов:** YOLOv8 (Ultralytics)
- **Оценка глубины:** Depth-Anything, DepthPro (HuggingFace Transformers)
- **Метрики:** SSIM, PSNR, FSIM, mAP

## Рабочий процесс

1. **Загрузка данных:**
   - Загрузка наборов изображений с аннотациями ограничивающих рамок
   - Преобразование меток классов в формат, совместимый с YOLO
   - Разделение данных на обучающую и валидационную выборки

2. **Генерация пятиканальных изображений:**
   - **Каналы RGB:** Исходные цветовые каналы
   - **Глубинный канал:** Генерируется с использованием предобученных моделей оценки глубины (Depth-Anything или DepthPro)
   - **Контурный канал:** Генерируется с использованием детектора границ Кэнни

3. **Архитектура модели:**
   - **Энкодер:** Многоуровневый сверточный энкодер с канальным вниманием
   - **Декодер:** Трехпоточный декодер (ветви цвета, глубины и контуров)
   - **Skip-соединения:** Распространение признаков от энкодера к декодеру

4. **Функции потерь:**
   - **YOLOv8 Loss:** Комбинированная функция потерь для детекции (box, cls, dfl)
   - **FOX Loss:** Многокомпонентная функция потерь для восстановления каналов
   - **RGBTargetAwareLoss:** Индивидуальные веса для каналов цвета, глубины и контуров
   - **Complex Loss:** Взвешенная комбинация потерь детекции и восстановления
   - **SSIM Loss:** Функция потерь структурного сходства с компонентом MSE

5. **Метрики:**
   - **PSNR:** Пиковое отношение сигнал-шум
   - **FSIM:** Индекс сходства признаков
   - **mAP:** Средняя точность для обнаружения объектов
   - **SSIM:** Индекс структурного сходства

## Установка

### 1. Предварительные требования

- Python 3.8 или выше
- GPU с поддержкой CUDA (рекомендуется)

### 2. Клонирование репозитория

```bash
git clone https://github.com/your_username/five-channel-image-processing.git
cd five-channel-image-processing
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Загрузка предобученных моделей

- YOLOv8x-oiv7.pt (предобученная модель детекции)
- Depth-Anything-V2-Small-hf (оценка глубины)
- DepthPro-hf (альтернативная модель оценки глубины)

### 5. Запуск обучения

```bash
python Main.py
```

## Детали архитектуры модели

### U-NET с Channel Attention

```
Вход (3-канальное RGB)
    ↓
Уровень энкодера 1: Conv2D(16) + MaxPool + ChannelAttention
    ↓
Уровень энкодера 2: Conv2D(32) + MaxPool + ChannelAttention
    ↓
Уровень энкодера 3: Conv2D(64) + MaxPool + ChannelAttention
    ↓
Уровень энкодера 4: Conv2D(128) + MaxPool + ChannelAttention
    ↓
Уровень декодера 1: UpConv + Skip-соединение (Ветвь цвета)
    ↓
Уровень декодера 2: UpConv + Skip-соединение (Ветвь глубины)
    ↓
Уровень декодера 3: UpConv + Skip-соединение (Ветвь контуров)
    ↓
Выход (3-канальное RGB восстановление)
```

## Результаты

### Визуальные результаты

**а) CDE 1 DA** | **б) CDE 2 DA** | **в) CDE 3 DA** | **г) CDE 3 DP** | **д) Исходное**

![CDE 1 DA](https://github.com/user-attachments/assets/5e8edcc4-111c-4041-aa73-f5c32f84f6ef) | ![CDE 2 DA](https://github.com/user-attachments/assets/12b5ceab-a81c-4a47-bb8f-aab7f39d1e47) | ![CDE 3 DA](https://github.com/user-attachments/assets/5444a6a5-6d3a-4730-9335-061c433ced3d) | ![CDE 3 DP](https://github.com/user-attachments/assets/d7e0ae22-919a-46cd-8111-4006c3b04cbb) | ![Original](https://github.com/user-attachments/assets/28b9136a-c80c-4d78-8cb4-4df8fe852bdd)

*Рисунок 1: Примеры восстановления пятиканальных изображений в трехканальные. а) CDE 1 DA, б) CDE 2 DA, в) CDE 3 DA, г) CDE 3 DP, д) Исходное изображение.*

### Результаты точности

![Результаты точности](https://github.com/user-attachments/assets/2f2fa563-7cd9-4a39-abbf-973ff39f0b06)

*Рисунок 2: Метрики точности модели на новых изображениях.*

## Выходные данные

Система генерирует следующие выходные данные:

1. **Восстановленные изображения:** Трехканальные изображения, восстановленные из пятиканального входа
2. **Результаты детекции:** Обнаружение объектов с ограничивающими рамками и метками классов
3. **Логи обучения:** Кривые потерь, гистограммы градиентов и метрики через WandB
4. **Веса модели:** Сохраненные контрольные точки модели для инференса
5. **Метрики:** Значения PSNR, SSIM, FSIM и mAP

## Перспективы развития

- Интеграция дополнительных моделей оценки глубины
- Реализация конвейера инференса в реальном времени
- Оптимизация для периферийных устройств
- Стратегия многоуровневого обучения
- Дополнительные методы аугментации данных
- Расширение на обработку видео
- Реализация дистилляции модели для ускорения инференса
