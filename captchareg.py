import cv2
import torch
import os


model_path = 'model/best.pt'
test_images_folder = '/test'
output_folder = '/output'

# Создаем папку для сохранения результатов, если она не существует
os.makedirs(output_folder, exist_ok=True)

# Определяем устройство: GPU, если доступен, иначе CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Загрузка модели YOLOv5 на соответствующее устройство
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(device)

# Переменная для подсчета обнаруженных букв
total_detected_letters = 0

# Проход по всем изображениям в папке
for image_name in os.listdir(test_images_folder):
    captcha_image_path = os.path.join(test_images_folder, image_name)

    # Загрузка изображения
    img = cv2.imread(captcha_image_path)

    # Проверка, что изображение загружено правильно
    if img is None:
        print(f"Ошибка: изображение '{image_name}' не загружено.")
        continue

    # Передача изображения в модель
    results = model(img)

    # Извлечение результатов детекции
    results = results.xyxy[0]  # [x1, y1, x2, y2, confidence, class]

    # Установка порога уверенности
    confidence_threshold = 0.5
    detections = results[results[:, 4] >= confidence_threshold]

    if detections is not None and len(detections) > 0:
        print(f"Обнаруженные объекты в '{image_name}':")
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(f"Координаты: ({x1}, {y1}), ({x2}, {y2}), Уверенность: {conf:.2f}, Класс: {model.names[int(cls)]}")

            # Увеличиваем счетчик обнаруженных букв
            total_detected_letters += 1

            # Рисуем рамки вокруг обнаруженных объектов
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Сохранение результата в файл
        output_path = os.path.join(output_folder, f"detected_{image_name}")
        cv2.imwrite(output_path, img)
        print(f"Результаты детекции сохранены в файл '{output_path}'.")
    else:
        print(f"Объекты не обнаружены в '{image_name}'.")

# Проверка на равенство 5 букв
if total_detected_letters > 0:
    detection_percentage = (total_detected_letters / 5) * 100
    print(f"Общее количество обнаруженных букв: {total_detected_letters}.")
    if total_detected_letters == 5:
        print("Процент обнаруженных букв от 5: 100.00%")
    else:
        print(f"Процент обнаруженных букв от 5: {detection_percentage:.2f}%")
else:
    print("Буквы не обнаружены.")
