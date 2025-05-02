import torch
import cv2
import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor
from Utils import draw_bbox
import os

sample_image = "temp_frame.jpg"

predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-base-plus")

def split_video(path, segment_duration=3):
        
    cap = cv2.VideoCapture(path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width =int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(f"FPS:{fps}")
    
    frame_size = (width, height)
    
    segment_frame_count =  segment_duration * fps
    current_segment = 0
    frame_count = 0
    
    
    if not cap.isOpened():
        print("Error opening file")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    
    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            break
        
        
        if frame_count % segment_frame_count == 0:
            if out:
                out.release()
            segment_path = f"./segments/segment_{current_segment}.mp4"
            out = cv2.VideoWriter(segment_path, fourcc = fourcc, fps = fps, frameSize=frame_size)
            first_frame = cv2.imread(sample_image)
            first_frame = cv2.resize(first_frame, (width, height))
            print(first_frame.shape, frame.shape)
            # for i in range(fps):
            out.write(first_frame)
            
            current_segment += 1
            
        out.write(frame)
        frame_count+=1

    # for _ in range(80):
    #     black_frame = 255*np.zeros((480, 640, 3), dtype = np.uint8)
    #     out.write(black_frame)
        
    out.release()
         
def run_detection_on_segments(segments_dir = "./segments"):
    
    os.makedirs(segments_dir, exist_ok=True)
    
    segment_paths = os.listdir(segments_dir)
    
    for path in segment_paths:
        
        video_path = f"{segments_dir}/{path}"
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # Initialize state with video
            state = predictor.init_state(video_path)

            # Add prompt for first frame
            frame_idx = 0
            points = np.array([[170, 384]])  # Click at (500, 300)
            labels = np.array([1])  # Foreground
            points = torch.from_numpy(points).to("cuda")
            labels = torch.from_numpy(labels).to("cuda")
            
            frame_idx, object_ids, masks = predictor.add_new_points(
                inference_state=state,
                frame_idx=frame_idx,
                points=points,
                labels=labels,
                obj_id = 1
            )
            print(f"Frame: {frame_idx}, Object IDs: {object_ids}, Mask shape: {masks.shape}")
            masks = masks.transpose(1, -1)
            masks = masks.transpose(1, 2)
            print(masks.shape)
            mask = masks[0].detach().cpu().numpy()  # Take the first mask, shape [H, W]
            mask = (mask > 0).astype(np.uint8) * 255  # Convert binary mask to 0-255 range

            print(f"Mask shape: {mask.shape}, Mask dtype: {mask.dtype}")
            # cv2.imshow('masks', mask)
            # cv2.waitKey(2)
            # Propagate across video
            for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
                print(f"Frame: {frame_idx}, Object IDs: {object_ids}, Mask shape: {masks.shape}")
                masks = masks.transpose(1, -1)
                masks = masks.transpose(1, 2)
                print(masks.shape)
                mask = masks[0].detach().cpu().numpy()  # Take the first mask, shape [H, W]
                mask = (mask > 0).astype(np.uint8) * 255  # Convert binary mask to 0-255 range
                cv2.imshow('masks', mask)
                cv2.waitKey(1)
                
                # cv2.imwrite(f"./masks/mask{frame_idx}.jpg", mask)
            cv2.destroyAllWindows()
    
    
def getframes(videopath):
    # global frames
    
    cam = cv2.VideoCapture(videopath)
    count = 0
    while True:
        count += 1
        ret, frame = cam.read()
        
        if not ret:
            break
        
        cv2.imwrite(f"./frames/frame{count}.jpg", frame)
        
    cam.release()
    
if __name__ == "__main__":
    split_video("Cat Short Sample Video.mp4")
    run_detection_on_segments()