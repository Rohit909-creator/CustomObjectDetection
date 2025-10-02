import torch
import cv2
import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor
from Utils import draw_bbox

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

def showframes():

    count = 0

    while True:

        count += 1

        mask = cv2.imread(f'./masks/mask{count}.jpg')
        frame = cv2.imread(f'./frames/frame{count}.jpg')

        draw_bbox(f'./masks/mask{count}.jpg', f'./frames/frame{count}.jpg')


getframes(videopath = "Object_detection_sample_test - Made with Clipchamp.mp4")

# Load model
predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-base-plus")

# Load and preprocess video
video_path = "output.mp4"
# Inference mode
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # Initialize state with video
    state = predictor.init_state(video_path)

    # Add prompt for first frame
    frame_idx = 0
    points = np.array([[346, 157]])  # Click at (500, 300)
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
        draw_bbox(mask, f'./SampleImages/00{frame_idx}.jpg')
        cv2.imshow('masks', mask)
        cv2.waitKey(1)
        cv2.imwrite(f"./masks/mask{frame_idx}.jpg", mask)
    cv2.destroyAllWindows()