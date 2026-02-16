"""
Fat Mex Nodes - Head/Object Mask
Wraps SAM3 model loading + grounding + mask growing into a single node.
"""

import numpy as np
import scipy.ndimage
import torch
import logging

import comfy.model_management

logger = logging.getLogger("FatMex")

_sam3_cache = {}


def _get_sam3_classes():
    """Get SAM3 node classes from ComfyUI's node registry (already loaded at startup)."""
    import nodes as comfy_nodes
    mappings = comfy_nodes.NODE_CLASS_MAPPINGS
    loader_cls = mappings.get("LoadSAM3Model")
    grounding_cls = mappings.get("SAM3Grounding")
    if loader_cls is None or grounding_cls is None:
        raise ImportError(
            "ComfyUI-SAM3 nodes not found. Please install ComfyUI-SAM3: "
            "https://github.com/neverbiasu/ComfyUI-SAM3"
        )
    return loader_cls, grounding_cls


def _get_sam3_model(model_path="models/sam3/sam3.pt"):
    """Load and cache SAM3 model. Reuses across executions."""
    if model_path in _sam3_cache:
        return _sam3_cache[model_path]

    loader_cls, _ = _get_sam3_classes()
    loader = loader_cls()
    result = loader.load_model(model_path)
    sam3_model = result[0]
    _sam3_cache[model_path] = sam3_model
    return sam3_model


def _get_sam3_grounding():
    """Get SAM3Grounding node instance."""
    _, grounding_cls = _get_sam3_classes()
    return grounding_cls()


def _grow_mask(mask, expand, tapered_corners=True):
    """Grow/shrink a mask by expand pixels. Replicates ComfyUI GrowMask logic."""
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                       [1, 1, 1],
                       [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)


class FatMexHeadMask:
    """
    All-in-one SAM3-based object masking.
    Replaces: LoadSAM3Model + SAM3Grounding + GrowMask + MaskPreview+

    Give it an image and a text prompt (default: 'hair. head. face.')
    and it returns a grown mask ready for inpainting.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to create a mask from."}),
            },
            "optional": {
                "mask_prompt": ("STRING", {
                    "default": "hair. head. face.",
                    "tooltip": "What to mask. SAM3 text grounding prompt. Separate items with periods."
                }),
                "grow_pixels": ("INT", {
                    "default": 15, "min": 0, "max": 100, "step": 1,
                    "tooltip": "Expand the mask by this many pixels (feathering)."
                }),
                "confidence": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "SAM3 detection confidence threshold."
                }),
                "sam3_model_path": ("STRING", {
                    "default": "models/sam3/sam3.pt",
                    "tooltip": "Path to SAM3 model (auto-downloads if missing)."
                }),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "preview")
    FUNCTION = "create_mask"
    CATEGORY = "Fat Mex"
    DESCRIPTION = (
        "All-in-one SAM3 head/object masking. Detects and masks objects matching "
        "the text prompt (default: full head). Returns a grown mask ready for "
        "inpainting, plus a visualization preview."
    )

    def create_mask(self, image, mask_prompt="hair. head. face.",
                    grow_pixels=15, confidence=0.2,
                    sam3_model_path="models/sam3/sam3.pt"):

        # Load SAM3 model (cached)
        sam3_model = _get_sam3_model(sam3_model_path)

        # Run SAM3 grounding
        grounding = _get_sam3_grounding()
        masks, visualization, boxes_json, scores_json = grounding.segment(
            sam3_model=sam3_model,
            image=image,
            confidence_threshold=confidence,
            text_prompt=mask_prompt,
        )

        # Grow the mask for feathering
        if grow_pixels > 0:
            masks = _grow_mask(masks, grow_pixels)

        return (masks, visualization)


MASK_CLASS_MAPPINGS = {
    "FatMexHeadMask": FatMexHeadMask,
}

MASK_NAME_MAPPINGS = {
    "FatMexHeadMask": "Fat Mex Head Mask",
}
