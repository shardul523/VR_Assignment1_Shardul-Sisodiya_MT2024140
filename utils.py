import os
import cv2

def save_image(image_path, image):
    output_dir = os.path.dirname(image_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(image_path, image)