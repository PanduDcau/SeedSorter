from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import Counter

# YOLO model path
YOLO_MODEL_PATH = 'models/best.pt'

# Load the trained YOLOv8 model
model = YOLO(YOLO_MODEL_PATH)


# Function to process image for brightness and sharpness
def process_image(file_path, result_folder='results'):
    os.makedirs(result_folder, exist_ok=True)

    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(result_folder, 'original.png'), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image_rgb, -1, sharpening_kernel)

    hsv_image = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], 50)
    brightened = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    cv2.imwrite(os.path.join(result_folder, 'brightened.png'), cv2.cvtColor(brightened, cv2.COLOR_RGB2BGR))

    return os.path.join(result_folder, 'brightened.png')


# Function to zoom into an image
def zoom_image(file_path, result_folder='results'):
    image = cv2.imread(file_path)
    h, w, _ = image.shape
    x_center, y_center = w // 2, h // 2
    zoom_factor = 2

    x_start = x_center - (w // (2 * zoom_factor))
    x_end = x_center + (w // (2 * zoom_factor))
    y_start = y_center - (h // (2 * zoom_factor))
    y_end = y_center + (h // (2 * zoom_factor))

    zoomed_image = image[y_start:y_end, x_start:x_end]
    zoomed_image = cv2.resize(zoomed_image, (w, h), interpolation=cv2.INTER_LINEAR)

    zoomed_image_path = os.path.join(result_folder, 'zoomed_image.jpg')
    cv2.imwrite(zoomed_image_path, zoomed_image)

    return zoomed_image_path


# Function to generate unique colors for each class
def generate_colors(num_classes):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors


# Function to display image with bounding boxes and calculate pepper seed purity
def display_image_with_boxes(image, results):
    num_classes = len(model.names)
    colors = generate_colors(num_classes)

    pepper_seed_class_id = None
    for idx, class_name in model.names.items():
        if "pepper" in class_name.lower():
            pepper_seed_class_id = idx
            break

    pepper_seed_count = 0
    total_count = 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls_id, confidence in zip(boxes, class_ids, confidences):
            x_min, y_min, x_max, y_max = box
            color = colors[cls_id].tolist()
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{model.names[cls_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            total_count += 1
            if cls_id == pepper_seed_class_id:
                pepper_seed_count += 1

    purity_percentage = (pepper_seed_count / total_count) * 100 if total_count > 0 else 0
    return image, purity_percentage, pepper_seed_count, total_count


# Function to count number of objects in each image
def count_objects(results):
    class_ids = []
    for result in results:
        class_ids.extend(result.boxes.cls.cpu().numpy().astype(int))
    count = Counter(class_ids)
    return sum(count.values()), count


# Process images in a folder and return detailed stats
def process_images_in_folder(folder_path):
    total_boxes = 0
    total_pepper_boxes = 0
    max_boxes_in_single_image = 0
    max_box_image_data = None

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        results = model(image)
        num_boxes, class_counts = count_objects(results)

        pepper_seed_class_id = None
        for idx, class_name in model.names.items():
            if "pepper" in class_name.lower():
                pepper_seed_class_id = idx
                break

        pepper_count = class_counts.get(pepper_seed_class_id, 0)

        total_boxes += num_boxes
        total_pepper_boxes += pepper_count

        if num_boxes > max_boxes_in_single_image:
            max_boxes_in_single_image = num_boxes
            max_box_image_data = (filename, image.copy(), results)

    return total_boxes, total_pepper_boxes, max_boxes_in_single_image, max_box_image_data


# Function to process the uploaded image sequentially
def process_uploaded_image(file_path, result_folder='results'):
    brightened_image_path = process_image(file_path, result_folder)
    zoomed_image_path = zoom_image(brightened_image_path, result_folder)

    results_folder = result_folder
    total_boxes_results, pepper_boxes_results, max_boxes_results, max_box_image_data_results = process_images_in_folder(
        results_folder)

    if max_box_image_data_results:
        filename, image, results = max_box_image_data_results
        image_with_boxes, purity, pepper_seed_count, total_count = display_image_with_boxes(image, results)
        output_path = os.path.join(result_folder, f'highest_boxes_{filename}')
        cv2.imwrite(output_path, image_with_boxes)
        return output_path, purity, pepper_seed_count, total_count
    return None, 0, 0, 0
