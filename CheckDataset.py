import cv2
from pathlib import Path

dataset_dir = Path("dataset")

for label_file in sorted((dataset_dir / "labels").glob("*.txt")):
    # Read label
    with open(label_file, 'r') as f:
        line = f.readline().strip().split()
        class_id, x_center, y_center, width, height = map(float, line)
    
    # Read corresponding image
    img_file = dataset_dir / "images" / label_file.name.replace('.txt', '.jpg')
    img = cv2.imread(str(img_file))
    
    if img is None:
        continue
    
    h, w = img.shape[:2]
    
    # Convert YOLO format to bbox coordinates
    x = int((x_center - width/2) * w)
    y = int((y_center - height/2) * h)
    box_w = int(width * w)
    box_h = int(height * h)
    
    # Draw bbox
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2)
    cv2.putText(img, f"Frame: {label_file.stem}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Dataset Viewer', img)
    if cv2.waitKey(2000) & 0xFF == 27:  # ESC to exit
        break

cv2.destroyAllWindows()