from ultralytics import YOLO
import cv2
import numpy as np
import os
from collections import Counter
import shutil

# YOLO model path
YOLO_MODEL_PATH = 'models/best.pt'

# Load the trained YOLOv8 model
model = YOLO(YOLO_MODEL_PATH)

# Ensure the result and process directories exist
result_folder = 'results'
results_process_folder = 'results_process'
output_folder = 'output'
os.makedirs(result_folder, exist_ok=True)
os.makedirs(results_process_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)


# Function to process image for brightness and sharpness
def process_image(file_path, result_folder='results'):
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(result_folder, 'original.png'), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image_rgb, -1, sharpening_kernel)

    hsv_image = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], 50)
    brightened = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    half = cv2.resize(brightened, (0, 0), fx=0.1, fy=0.1)
    bigger = cv2.resize(brightened, (1050, 1610))
    stretch_near = cv2.resize(brightened, (780, 540), interpolation=cv2.INTER_LINEAR)

    # Titles and images for display
    images = [image_rgb, half, bigger, stretch_near]
    image_names = ["original.png", "half.png", "bigger.png", "stretch_near.png"]
    count = 4
    for i in range(count):
        cv2.imwrite(os.path.join(result_folder, image_names[i]), cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))

    return os.path.join(result_folder, image_names[i])


# Function to zoom into an image and save in results_process
def zoom_image(file_path='results/bigger.png', result_folder='results_process'):
    image = cv2.imread(file_path)
    h, w, _ = image.shape
    x_center, y_center = w // 2, h // 2
    zoom_factor = 2

    x_start = x_center - (w // (2 * zoom_factor))
    x_end = x_center + (w // (2 * zoom_factor))
    y_start = y_center - (h // (2 * zoom_factor))
    y_end = y_center + (h // (2 * zoom_factor))

    # zoomed_image = image[y_start:y_end, x_start:x_end]
    # zoomed_image = cv2.resize(zoomed_image, (w, h), interpolation=cv2.INTER_LINEAR)
    #
    # zoomed_image_path = os.path.join(result_folder, 'zoomed_image.jpg')
    # cv2.imwrite(zoomed_image_path, zoomed_image)

    file_path = 'results/original.png'  # Replace with your actual file path
    image = cv2.imread(file_path)

    # Get the dimensions of the image
    h, w, _ = image.shape

    # Calculate new dimensions after zoom
    new_h, new_w = h // zoom_factor, w // zoom_factor

    # Define the coordinates for the four quadrants
    coordinates = [
        (0, new_h, 0, new_w),  # Top-left
        (0, new_h, new_w, w),  # Top-right
        (new_h, h, 0, new_w),  # Bottom-left
        (new_h, h, new_w, w)  # Bottom-right
    ]

    # Process each quadrant
    for i, (y_start, y_end, x_start, x_end) in enumerate(coordinates):
        # Crop the quadrant from the image
        cropped_image = image[y_start:y_end, x_start:x_end]

        # Resize the cropped image to original size for better visibility
        zoomed_image = cv2.resize(cropped_image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Save the zoomed image
        zoomed_image_path = os.path.join(result_folder, f'zoomed_image_{i + 1}.jpg')
        cv2.imwrite(zoomed_image_path, zoomed_image)

    return zoomed_image_path


# Function to generate unique colors for each class
def generate_colors(num_classes):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors


# Function to display image with bounding boxes and calculate pepper seed purity
def display_image_with_boxes(image, results, pepper_seed_class_id):
    num_classes = len(model.names)
    colors = generate_colors(num_classes)

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
def process_images_in_folder(folder_path, pepper_seed_class_id):
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

        pepper_count = class_counts.get(pepper_seed_class_id, 0)

        total_boxes += num_boxes
        total_pepper_boxes += pepper_count

        if num_boxes > max_boxes_in_single_image:
            max_boxes_in_single_image = num_boxes
            max_box_image_data = (filename, image.copy(), results)

    return total_boxes, total_pepper_boxes, max_boxes_in_single_image, max_box_image_data

def display_image_box(image, results, pepper_seed_class_id):
    num_classes = len(model.names)
    colors = generate_colors(num_classes)

    pepper_seed_count = 0
    total_count = 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # Extract boxes in (xmin, ymin, xmax, ymax) format
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        # Draw bounding boxes and count objects
        for box, cls_id, confidence in zip(boxes, class_ids, confidences):
            x_min, y_min, x_max, y_max = box
            color = colors[cls_id].tolist()  # Assign a unique color for each class
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{model.names[cls_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Count pepper seeds and total objects
            total_count += 1
            if cls_id == pepper_seed_class_id:
                pepper_seed_count += 1
    return image


def process_uploaded_image(file_path, result_folder='results'):

    #First process and brighten the image
    brightened_image_path = process_image(file_path, result_folder)

    # Zoom into the image and save it to the results_process folder
    zoomed_image_path = zoom_image(file_path, results_process_folder)

    total_boxes_results, pepper_boxes_results, max_boxes_results, max_box_image_data_results = process_images_in_folder(
        result_folder, find_pepper_seed_class_id())
    print(f"Total bounding boxes in Processed Images: {total_boxes_results}")
    print(f"Total pepper seed bounding boxes in Processed Images: {pepper_boxes_results}")
    print(f"Highest number of bounding boxes in a single image in 'results': {max_boxes_results}")

    # Process images in results_zoomed folder
    total_boxes_zoomed, pepper_boxes_zoomed, max_boxes_zoomed, max_box_image_data_zoomed = process_images_in_folder(
        results_process_folder, find_pepper_seed_class_id())

    print(f"Total bounding boxes in Segmented Image: {total_boxes_zoomed}")
    print(f"Total pepper seed bounding boxes in Segmented Image: {pepper_boxes_zoomed}")
    print(f"Highest number of bounding boxes in a single image in Segmented Image: {max_boxes_zoomed}")

    # Determine the folder with the highest bounding boxes for purity calculation
    if max_boxes_results > total_boxes_zoomed:
        total_boxes = max_boxes_results
        pepper_boxes = max_box_image_data_results[2][0].boxes.cls.cpu().numpy().tolist().count(find_pepper_seed_class_id())
        folder_name = 'results'
    else:
        total_boxes = total_boxes_zoomed
        pepper_boxes = pepper_boxes_zoomed
        folder_name = 'results_process'

    # Calculate the pepper seed purity
    pepper_seed_purity = (pepper_boxes / total_boxes) * 100 if total_boxes > 0 else 0
    print(f"Pepper Seed Purity from '{folder_name}': {pepper_seed_purity:.2f}% ({pepper_boxes}/{total_boxes})")
    print(f"Pepper Seeds Count in the Sample: {pepper_boxes}")

    # Save the image with the highest number of bounding boxes
    filename, image, results = max_box_image_data_results \
        if max_boxes_results > total_boxes_zoomed else max_box_image_data_zoomed
    image_with_boxes, purity, pepper_seed_count, total_count = display_image_with_boxes(image, results, find_pepper_seed_class_id())
    output_path = os.path.join(output_folder, f'highest_boxes_{filename}')
    cv2.imwrite(output_path, image_with_boxes)
    print(f"Image saved at: {output_path}")
    return output_path, pepper_seed_purity, pepper_boxes_zoomed, total_boxes_zoomed

# Helper function to find the pepper seed class id
def find_pepper_seed_class_id():
    pepper_seed_class_id = None
    for idx, class_name in model.names.items():
        if "pepper" in class_name.lower():
            pepper_seed_class_id = idx
            break
    return pepper_seed_class_id
