"""
Fat Mex Nodes - Dataset Saver
Save images in a dataset-ready folder structure with auto-naming and optional captioning.
"""

import os
import json
import logging
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from comfy.cli_args import args

logger = logging.getLogger("FatMex")


class FatMexDatasetSaver:
    """
    Save images to a dataset-ready folder with sequential naming and optional captions.
    Replaces: SaveImage + SaveImageDataSetToFolder + manual folder setup
    """

    def __init__(self):
        self.counter = 0

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Images to save to the dataset folder."}),
                "dataset_name": ("STRING", {"default": "my_dataset", "tooltip": "Name of the dataset folder."}),
                "filename_prefix": ("STRING", {"default": "img", "tooltip": "Prefix for each image file."}),
            },
            "optional": {
                "caption": ("STRING", {"default": "", "multiline": True, "tooltip": "Caption/description. Saved as .txt alongside each image."}),
                "output_format": (["png", "jpg", "webp"], {"default": "png", "tooltip": "Image format."}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "JPEG/WebP quality (ignored for PNG)."}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 999999, "tooltip": "Starting index for file numbering."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path",)
    OUTPUT_NODE = True
    FUNCTION = "save_dataset"
    CATEGORY = "Fat Mex"
    DESCRIPTION = "Save images to a dataset folder with sequential numbering and optional caption .txt files. Perfect for building training datasets."

    def save_dataset(self, images, dataset_name, filename_prefix,
                     caption="", output_format="png", jpg_quality=95,
                     start_index=0, prompt=None, extra_pnginfo=None):

        # Create dataset directory
        output_base = folder_paths.get_output_directory()
        dataset_dir = os.path.join(output_base, "datasets", dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        # Find next available index if not specified
        if start_index > 0:
            counter = start_index
        else:
            existing = [f for f in os.listdir(dataset_dir) if f.startswith(filename_prefix)]
            counter = len(existing)

        results = []
        saved_paths = []

        for batch_idx, image in enumerate(images):
            # Convert tensor to PIL
            img_np = (255.0 * image.cpu().numpy()).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(img_np)

            # Generate filename
            idx = counter + batch_idx
            base_name = f"{filename_prefix}_{idx:05d}"

            # Save image
            if output_format == "png":
                filepath = os.path.join(dataset_dir, f"{base_name}.png")
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for key in extra_pnginfo:
                            metadata.add_text(key, json.dumps(extra_pnginfo[key]))
                pil_image.save(filepath, pnginfo=metadata)
            elif output_format == "jpg":
                filepath = os.path.join(dataset_dir, f"{base_name}.jpg")
                pil_image.save(filepath, quality=jpg_quality)
            elif output_format == "webp":
                filepath = os.path.join(dataset_dir, f"{base_name}.webp")
                pil_image.save(filepath, quality=jpg_quality)

            saved_paths.append(filepath)

            # Save caption if provided
            if caption.strip():
                caption_path = os.path.join(dataset_dir, f"{base_name}.txt")
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(caption.strip())

            results.append({
                "filename": os.path.basename(filepath),
                "subfolder": os.path.join("datasets", dataset_name),
                "type": "output"
            })

        self.counter = counter + len(images)
        logger.info(f"FatMex: Saved {len(images)} images to {dataset_dir}")

        return {"ui": {"images": results}, "result": (dataset_dir,)}


class FatMexImageCompare:
    """
    Side-by-side image comparison viewer.
    Takes two images and stitches them horizontally for easy comparison.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "First image (left side)."}),
                "image_b": ("IMAGE", {"tooltip": "Second image (right side)."}),
            },
            "optional": {
                "label_a": ("STRING", {"default": "Before", "tooltip": "Label for image A."}),
                "label_b": ("STRING", {"default": "After", "tooltip": "Label for image B."}),
                "gap": ("INT", {"default": 4, "min": 0, "max": 50, "tooltip": "Gap between images in pixels."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("comparison",)
    FUNCTION = "compare"
    CATEGORY = "Fat Mex"
    DESCRIPTION = "Create a side-by-side comparison of two images."

    def compare(self, image_a, image_b, label_a="Before", label_b="After", gap=4):
        import torch

        # Get first image from each batch
        a = image_a[0] if len(image_a.shape) == 4 else image_a
        b = image_b[0] if len(image_b.shape) == 4 else image_b

        h_a, w_a, c_a = a.shape
        h_b, w_b, c_b = b.shape

        # Match heights
        target_h = max(h_a, h_b)
        if h_a != target_h:
            scale = target_h / h_a
            new_w = int(w_a * scale)
            a_pil = Image.fromarray((a.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            a_pil = a_pil.resize((new_w, target_h), Image.LANCZOS)
            a = torch.from_numpy(np.array(a_pil).astype(np.float32) / 255.0)
            w_a = new_w

        if h_b != target_h:
            scale = target_h / h_b
            new_w = int(w_b * scale)
            b_pil = Image.fromarray((b.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
            b_pil = b_pil.resize((new_w, target_h), Image.LANCZOS)
            b = torch.from_numpy(np.array(b_pil).astype(np.float32) / 255.0)
            w_b = new_w

        # Create gap
        gap_tensor = torch.ones(target_h, gap, 3) * 0.2  # Dark gray gap

        # Stitch
        combined = torch.cat([a, gap_tensor, b], dim=1)

        return (combined.unsqueeze(0),)


DATASET_CLASS_MAPPINGS = {
    "FatMexDatasetSaver": FatMexDatasetSaver,
    "FatMexImageCompare": FatMexImageCompare,
}

DATASET_NAME_MAPPINGS = {
    "FatMexDatasetSaver": "Fat Mex Dataset Saver",
    "FatMexImageCompare": "Fat Mex Image Compare",
}
