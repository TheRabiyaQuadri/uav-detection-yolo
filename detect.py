
import torch
import models.yolo  # Import the module containing the class

# Allow PyTorch to safely unpickle the YOLO Model class
torch.serialization.add_safe_globals([models.yolo.Model])
import cv2
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
# With standard importlib metadata:
import importlib.metadata as pkg


def get_model(yolo_weights_file, model_image_size=640):
    """Initializes and returns the YOLO detection model with appropriate precision."""
    device = select_device("")
    model = DetectMultiBackend(yolo_weights_file, device=device)
    imgsz = check_img_size(model_image_size, s=model.stride)

    print(f"Inference Device: {device.type} | Resolution: {imgsz}x{imgsz}")

    if model.pt and device.type != "cpu":
        print("FP16 Half-Precision Activated for GPU")
        model.model.half()
        # Warmup forward pass
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.model.parameters())))
    else:
        model.model.float()

    print("Model ready for inference.")
    return model, imgsz, device

import cv2
import torch
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


def detect_video_stream(model_yolo, video_path, output_path="output_detection.mp4", conf_thres=0.60, iou_thres=0.70):
    model, imgsz, device = model_yolo

    # 1. Open input video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 2. Setup OpenCV VideoWriter with H.264 codec for macOS compatibility
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video ({total_frames} total frames at {fps} FPS)...")

    frame_count = 0

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break  # End of video stream

        frame_count += 1

        # Preprocess frame (Resize & Letterbox)
        img = cv2.resize(raw_frame, (imgsz, imgsz))
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
        img = torch.from_numpy(img.copy()).to(device)
        img = img.half() / 255.0 if device.type != "cpu" else img.float() / 255.0

        if len(img.shape) == 3:
            img = img[None]  # Add batch dimension [1, 3, 640, 640]

        # Inference & Non-Max Suppression
        pred = non_max_suppression(model(img), conf_thres, iou_thres, max_det=100)

        # Annotate detections
        annotator = Annotator(raw_frame, line_width=2, example=str(model.names))

        for det in pred:
            if len(det):
                # Rescale boxes to raw frame dimensions
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], raw_frame.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    cls = int(cls)
                    label = f"{model.names[cls]} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=colors(cls, True))

        # Render annotated frame
        annotated_frame = annotator.result()

        # Write current frame to video
        writer.write(annotated_frame)

        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"Processed frame {frame_count}/{total_frames}")

    # 3. Release resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"Done! Full video saved to: {output_path}")


if __name__ == "__main__":
    # Load model
    model_yolo = get_model("weights/uav_model.pt", model_image_size=640)

    # Run on test video
    detect_video_stream(model_yolo, video_path="sample_video.mp4", output_path="output_test.mp4")