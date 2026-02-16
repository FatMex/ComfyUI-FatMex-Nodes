"""
Fat Mex Nodes - Custom ComfyUI Node Pack
All-in-one nodes for AI influencer content creation.

Nodes included:
  - Fat Mex Model Loader: Load any supported model with one node
  - Fat Mex Content Sampler: Text-to-image in one node
  - Fat Mex Reference Sampler: Reference-guided generation
  - Fat Mex Image Edit Sampler: Qwen image edit in one node
  - Fat Mex Inpaint Sampler: Masked inpainting with image references
  - Fat Mex Head Mask: SAM3-based head/object masking
  - Fat Mex Upscaler: All-in-one image upscaling
  - Fat Mex Prompt Batch: Multi-prompt batch encoding
  - Fat Mex Prompt Template: AI influencer prompt generator
  - Fat Mex Face Swap: One-click face swap
  - Fat Mex Dataset Saver: Save to dataset folders
  - Fat Mex Image Compare: Side-by-side comparison
"""

from .nodes_loader import LOADER_CLASS_MAPPINGS, LOADER_NAME_MAPPINGS
from .nodes_sampler import SAMPLER_CLASS_MAPPINGS, SAMPLER_NAME_MAPPINGS
from .nodes_prompt import PROMPT_CLASS_MAPPINGS, PROMPT_NAME_MAPPINGS
from .nodes_faceswap import FACESWAP_CLASS_MAPPINGS, FACESWAP_NAME_MAPPINGS
from .nodes_dataset import DATASET_CLASS_MAPPINGS, DATASET_NAME_MAPPINGS
from .nodes_mask import MASK_CLASS_MAPPINGS, MASK_NAME_MAPPINGS
from .nodes_upscale import UPSCALE_CLASS_MAPPINGS, UPSCALE_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_CLASS_MAPPINGS.update(LOADER_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(LOADER_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(SAMPLER_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(SAMPLER_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(PROMPT_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(PROMPT_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(FACESWAP_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(FACESWAP_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(DATASET_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(DATASET_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(MASK_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(MASK_NAME_MAPPINGS)

NODE_CLASS_MAPPINGS.update(UPSCALE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(UPSCALE_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print(f"\033[92m[Fat Mex Nodes] Loaded {len(NODE_CLASS_MAPPINGS)} nodes successfully\033[0m")
