# Utils.py
import cv2
import numpy as np
from pathlib import Path
import os

def click_image(image_path):
    """Allow user to click on object to select ROI"""
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    image = cv2.resize(image, (640, 480))
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, str(len(points)), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Select Object", image)
    
    cv2.imshow("Select Object", image)
    cv2.setMouseCallback("Select Object", mouse_callback)
    
    print("\nInstructions:")
    print("  - Left click to select object point(s)")
    print("  - Press SPACE or ENTER when done")
    print("  - Press ESC to cancel\n")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in [13, 32]:  # Enter or Space
            break
        elif key == 27:  # ESC
            points = []
            break
    
    cv2.destroyAllWindows()
    return points


def folder2video(folder_path, output_path, fps=30, target_size=(640, 480)):
    """Convert folder of images to video"""
    
    images = sorted(Path(folder_path).glob("*.[jp][pn][g]"))
    if not images:
        raise ValueError(f"No images found in {folder_path}")
    
    w, h = target_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.resize(img, (w, h))
            writer.write(img)
    
    writer.release()
    return (h, w)


def draw_bbox(mask, image_path, output_path):
    """Draw bounding box on image from mask"""
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame = cv2.imread(image_path)
    
    if frame is None:
        return None
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imwrite(output_path, frame)
    return (x, y, w, h)


def prepare_dataset(annotations_dict, output_dir):
    """Helper to organize dataset in YOLO format"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    (output_dir / "labels").mkdir(exist_ok=True)
    
def rename_files_in_folder(folder_path):

    for i, filename in enumerate(os.listdir(folder_path)):

        old_file = os.path.join(folder_path, filename)

        if os.path.isdir(old_file):
            continue

        new_filename = filename.replace(filename, f"00{i}.jpg")
        new_file = os.path.join(folder_path, new_filename)
        os.rename(old_file, new_file)

        print(f"renamed: {filename} to {new_filename}")