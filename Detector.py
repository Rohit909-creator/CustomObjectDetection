# main.py
import torch
import cv2
import numpy as np
import json
from pathlib import Path
from sam2.sam2_video_predictor import SAM2VideoPredictor
from Utils import click_image, draw_bbox, folder2video, prepare_dataset

class ObjectDetectionDatasetCreator:
    def __init__(self, image_folder, model_name="facebook/sam2-hiera-base-plus"):
        self.image_folder = Path(image_folder)
        self.temp_video = "temp_video.mp4"
        self.output_dir = Path("dataset")
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for YOLO format
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        
        self.predictor = SAM2VideoPredictor.from_pretrained(model_name)
        self.annotations = {}
        
    def create_dataset(self):
        """Main pipeline to create object detection dataset"""
        
        # Step 1: Convert images to video
        print("Converting images to video...")
        images = sorted(self.image_folder.glob("*.[jp][pn][g]"))
        if not images:
            raise ValueError(f"No images found in {self.image_folder}")
        
        img_shape = folder2video(str(self.image_folder), self.temp_video)
        print(f"Video created: {self.temp_video}")
        
        # Step 2: Get user click for ROI selection
        print("\nClick on the object you want to track in the first image...")
        first_image = str(images[0])
        points = click_image(first_image)
        
        if points is None or len(points) == 0:
            raise ValueError("No points selected. Please click on the object.")
        
        print(f"Selected points: {points}")
        
        # Step 3: Run SAM2 tracking
        print("\nRunning object tracking...")
        self._track_and_save(points, img_shape)
        
        # Step 4: Save annotations in YOLO format
        print("\nSaving annotations...")
        self._save_yolo_format(img_shape)
        
        # Cleanup
        Path(self.temp_video).unlink(missing_ok=True)
        
        print(f"\n✓ Dataset created successfully in '{self.output_dir}' directory!")
        print(f"  - Images: {len(self.annotations)} files")
        print(f"  - Format: YOLO (images/ and labels/ folders)")
        
    def _track_and_save(self, points, img_shape):
        """Track object through video and extract bounding boxes"""
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.predictor.init_state(self.temp_video)
            
            # Prepare points for SAM2
            points_array = np.array(points)
            labels_array = np.array([1] * len(points))  # All foreground
            points_tensor = torch.from_numpy(points_array).to("cuda")
            labels_tensor = torch.from_numpy(labels_array).to("cuda")
            
            # Add initial points
            frame_idx, object_ids, masks = self.predictor.add_new_points(
                inference_state=state,
                frame_idx=0,
                points=points_tensor,
                labels=labels_tensor,
                obj_id=1
            )
            
            self._process_mask(0, masks, img_shape)
            
            # Propagate through video
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(state):
                self._process_mask(frame_idx, masks, img_shape)
                print(f"Processed frame {frame_idx + 1}", end="\r")
            
            print()  # New line after progress
    
    def _process_mask(self, frame_idx, masks, img_shape):
        """Process mask and extract bounding box"""
        
        # Convert mask to numpy
        mask = masks[0].squeeze().detach().cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Store annotation
            image_name = f"image_{frame_idx:04d}.jpg"
            self.annotations[image_name] = {
                'bbox': [x, y, w, h],
                'frame_idx': frame_idx
            }
            
            # Copy image to output directory
            images = sorted(self.image_folder.glob("*.[jp][pn][g]"))
            if frame_idx < len(images):
                src_img = cv2.imread(str(images[frame_idx]))
                cv2.imwrite(str(self.output_dir / "images" / image_name), src_img)
    
    def _save_yolo_format(self, img_shape):
        """Save annotations in YOLO format"""
        
        h, w = img_shape
        
        for image_name, data in self.annotations.items():
            x, y, box_w, box_h = data['bbox']
            
            # Convert to YOLO format (normalized center x, y, width, height)
            x_center = (x + box_w / 2) / w
            y_center = (y + box_h / 2) / h
            norm_width = box_w / w
            norm_height = box_h / h
            
            # Save label file (class_id x_center y_center width height)
            label_name = image_name.replace('.jpg', '.txt')
            label_path = self.output_dir / "labels" / label_name
            
            with open(label_path, 'w') as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        # Save dataset.yaml for YOLO
        yaml_content = f"""# Dataset configuration for YOLO training
path: {self.output_dir.absolute()}
train: images
val: images

nc: 1  # number of classes
names: ['object']  # class names
"""
        with open(self.output_dir / "dataset.yaml", 'w') as f:
            f.write(yaml_content)
        
        # Also save raw annotations as JSON for reference
        with open(self.output_dir / "annotations.json", 'w') as f:
            json.dump(self.annotations, f, indent=2)


if __name__ == "__main__":
    # Simple usage
    creator = ObjectDetectionDatasetCreator(
        image_folder="./SampleImages"
    )
    creator.create_dataset()