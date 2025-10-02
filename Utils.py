import cv2
import numpy as np
import cv2
import os
from pathlib import Path


def click_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))
    if image is None:
        print("Failed to load image. Check the path.")
        return

    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Left click at: ({x}, {y})")
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Optional: show click
            cv2.imshow("Click on Image", image)

        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f"Right click at: ({x}, {y})")

    # Create window and bind callback
    cv2.imshow("Click on Image", image)
    cv2.setMouseCallback("Click on Image", mouse_callback)

    # Wait until any key is pressed
    print("Click on the image. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_bbox(img, output_img_path, save_dir="./Training_data"):
    
    # os.mkdir(save_dir)
    
    # Load the segmented image
    # img = cv2.imread(segmented_img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or could not be loaded.")
    
    # Find contours of the white object
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Load original image in color to draw bounding box
    # img_color = cv2.imread(segmented_img_path)
    frame = cv2.imread(output_img_path)
    # Draw bounding boxes around detected objects
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Save the output image
    save_path = os.path.join(save_dir, output_img_path)
    cv2.imwrite(save_path, frame)
    return output_img_path, (x,y,w,h)

def folder2Video(path):
    output = "output.mp4"
    fps = 1

    images = sorted(Path(path).glob("*.[jp][pn][g]"))
    frame = cv2.imread(str(images[0]))
    h, w = 480, 640

    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.resize(img, (w, h))
            writer.write(img)

    writer.release()

def rename_files_in_folder(folder_path):
    
    for i, filename in enumerate(os.listdir(folder_path)):
        
        old_file = os.path.join(folder_path, filename)
        
        if os.path.isdir(old_file):
            continue
        
        new_filename = filename.replace(filename, f"00{i}.jpg")
        new_file = os.path.join(folder_path, new_filename)
        os.rename(old_file, new_file)
        
        print(f"renamed: {filename} to {new_filename}")

if __name__ == "__main__":
    # Example usage
    click_image("./SampleImages/000.jpg")  # Replace with your image path
    # path = "./SampleImages"
    # rename_files_in_folder(path)
    # folder2Video(path)
    
# draw_bbox(f'./masks/mask188.jpg', f'./frames/frame188.jpg')
