import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, List, Dict, Literal
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class Grounding_SAM2:
    def __init__(
        self, 
        sam2_checkpoint: str, 
        sam2_config: str, 
        grounding_model_id: str, 
        device: torch.device
    ):
        self.device = device
        
        # Initialize SAM 2
        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Initialize Grounding DINO
        self.processor = AutoProcessor.from_pretrained(grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(grounding_model_id).to(device)

    def infer_image(
        self, 
        image: Image.Image, 
        prompt: str, 
        box_threshold: float, 
        text_threshold: float
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Given an image and prompt, returns masks and their corresponding text labels.
        Returns (None, []) if no objects are detected.
        """
        # 1. Detection (Grounding DINO)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.grounding_model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]]
        )[0]

        if len(results["boxes"]) == 0:
            return None, []

        # 2. Segmentation (SAM 2)
        self.sam2_predictor.set_image(np.array(image))
        masks, _, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=results["boxes"].cpu().numpy(),
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)  # Shape: [N, H, W]

        return masks, results["text_labels"]


def setup_environment(device_name: Optional[str] = None) -> torch.device:
    """Sets up torch device and performance flags."""
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    print(f"Using device: {device}")
    return device


def get_label_mapping(prompt: str) -> Dict[str, int]:
    """Creates a mapping from tags to unique integer IDs."""
    # Cleans prompt and splits by space to get unique categories
    unique_tags = prompt.strip().replace('.', '').split(' ')
    return {tag: i + 1 for i, tag in enumerate(unique_tags)}


def parse_args():
    parser = argparse.ArgumentParser(description="Grounded SAM 2 Image Segmentation")
    parser.add_argument("--data_dir", type=str, default="data/dynerf/cook_spinach/time0")
    parser.add_argument("--output_dir", type=str, default="data/dynerf/cook_spinach/object_mask")
    parser.add_argument("--sam2_checkpoint", type=str, default="submodules/sam2/checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_t.yaml")
    parser.add_argument("--grounding_model", type=str, default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--prompt", type=str, default="guy. bottle.")
    parser.add_argument("--box_threshold", type=float, default=0.4)
    parser.add_argument("--text_threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = setup_environment(args.device)
    
    # Initialize the pipeline class
    gsm2 = Grounding_SAM2(
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        grounding_model_id=args.grounding_model,
        device=device
    )

    tag2id = get_label_mapping(args.prompt)
    dataset_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # # Filter directories (e.g., cam00, cam01)
    # cam_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith("cam")]
    
    # Use bfloat16 context if supported
    autocast_type = device.type if device.type != "cpu" else "cpu"
    
    with torch.autocast(device_type=autocast_type, dtype=torch.bfloat16):
        # for sub_path in cam_dirs:
        #     images_dir = sub_path / "images"
        #     if not images_dir.exists():
        #         continue
                
        #     image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
        #     if not image_paths:
        #         continue
                
        #     # Process first frame
        #     first_frame_path = image_paths[0]
        for image_path in dataset_path.glob(f"*.jpg"):
            print(f"Processing frame: {image_path.name}")
            image_pil = Image.open(image_path).convert("RGB")
            
            # Inference
            masks, labels = gsm2.infer_image(
                image=image_pil,
                prompt=args.prompt,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold
            )
            
            if masks is None:
                print(f"No objects detected for {image_path.name}")
                continue

            # Create Semantic Label Map
            # masks shape is (N, H, W)
            label_image = np.zeros(masks.shape[1:], dtype=np.uint8)
            for mask, tag in zip(masks, labels):
                obj_id = tag2id.get(tag, 0)
                label_image[mask > 0.0] = obj_id
            
            # Save result
            saving_path = output_path / f"{image_path.stem}.png"
            cv2.imwrite(str(saving_path), label_image)
            print(f"Saved: {saving_path}")


if __name__ == "__main__":
    main()