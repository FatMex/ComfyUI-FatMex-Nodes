"""
Fat Mex Nodes - Prompt Batch
Enhanced prompt list with built-in CLIP encoding for batch generation.
"""

import logging

logger = logging.getLogger("FatMex")


class FatMexPromptBatch:
    """
    Multi-prompt batch encoder.
    Replaces: CR Prompt List + CLIPTextEncode
    Enter multiple prompts (one per line) and get them encoded sequentially.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The CLIP model for encoding prompts."}),
                "prompts": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "a beautiful woman in a red dress, professional photo\na woman in casual wear at a cafe, natural lighting\na woman at the beach, golden hour photography",
                    "tooltip": "One prompt per line. Each line generates a separate image."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Negative prompt applied to ALL images in the batch."
                }),
            },
            "optional": {
                "prefix": ("STRING", {"default": "", "tooltip": "Text prepended to every prompt."}),
                "suffix": ("STRING", {"default": "", "tooltip": "Text appended to every prompt."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("positive_batch", "negative", "count", "prompt_list")
    OUTPUT_IS_LIST = (True, False, False, True)
    OUTPUT_TOOLTIPS = (
        "List of encoded positive conditionings (one per prompt line).",
        "Single encoded negative conditioning (shared).",
        "Number of prompts in the batch.",
        "List of the raw prompt strings.",
    )
    FUNCTION = "encode_batch"
    CATEGORY = "Fat Mex"
    DESCRIPTION = "Encode multiple prompts at once for batch generation. Enter one prompt per line. Optionally add a prefix/suffix to all prompts."

    def encode_batch(self, clip, prompts, negative_prompt, prefix="", suffix=""):
        # Parse prompt lines
        lines = [line.strip() for line in prompts.strip().split("\n") if line.strip()]

        if not lines:
            raise ValueError("FatMex Prompt Batch: No prompts provided. Enter at least one prompt.")

        # Apply prefix/suffix
        processed = []
        for line in lines:
            full_prompt = f"{prefix} {line} {suffix}".strip()
            processed.append(full_prompt)

        # Encode each positive prompt
        positive_list = []
        for prompt_text in processed:
            tokens = clip.tokenize(prompt_text)
            cond = clip.encode_from_tokens_scheduled(tokens)
            positive_list.append(cond)

        # Encode negative (shared)
        neg_tokens = clip.tokenize(negative_prompt)
        negative = clip.encode_from_tokens_scheduled(neg_tokens)

        count = len(processed)
        logger.info(f"FatMex: Encoded {count} prompts for batch generation")

        return (positive_list, negative, count, processed)


class FatMexPromptTemplate:
    """
    AI Influencer prompt template generator.
    Produces structured prompts for influencer-style content.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "description": ("STRING", {
                    "multiline": True,
                    "default": "woman in red bikini at the beach",
                    "tooltip": "Brief description of the scene/outfit."
                }),
                "style": ([
                    "Instagram Photo",
                    "Professional Portrait",
                    "Casual Selfie",
                    "Fashion Editorial",
                    "Fitness Photo",
                    "Lifestyle Candid",
                    "Glamour Shot",
                    "Street Style",
                ], {"default": "Instagram Photo", "tooltip": "Photography style to apply."}),
                "quality_boost": ("BOOLEAN", {"default": True, "tooltip": "Add quality-enhancing terms to the prompt."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Fat Mex"
    DESCRIPTION = "Generate optimized prompts for AI influencer content. Pick a style and describe the scene."

    STYLE_TEMPLATES = {
        "Instagram Photo": "instagram photo, {desc}, aesthetic, well-lit, high quality smartphone photo",
        "Professional Portrait": "professional portrait photography, {desc}, studio lighting, bokeh background, 85mm lens",
        "Casual Selfie": "casual selfie, {desc}, natural lighting, slightly imperfect, authentic feel",
        "Fashion Editorial": "high fashion editorial, {desc}, dramatic lighting, vogue style, haute couture",
        "Fitness Photo": "fitness photography, {desc}, athletic, gym or outdoor setting, dynamic pose, motivational",
        "Lifestyle Candid": "candid lifestyle photo, {desc}, natural moment, warm tones, storytelling",
        "Glamour Shot": "glamour photography, {desc}, beautiful lighting, alluring pose, high-end",
        "Street Style": "street style photography, {desc}, urban background, fashionable, editorial street photo",
    }

    QUALITY_TERMS = ", masterpiece, best quality, highly detailed, sharp focus, 8k uhd, professional photography"

    def generate_prompt(self, description, style, quality_boost):
        template = self.STYLE_TEMPLATES.get(style, "{desc}")
        prompt = template.format(desc=description)

        if quality_boost:
            prompt += self.QUALITY_TERMS

        return (prompt,)


PROMPT_CLASS_MAPPINGS = {
    "FatMexPromptBatch": FatMexPromptBatch,
    "FatMexPromptTemplate": FatMexPromptTemplate,
}

PROMPT_NAME_MAPPINGS = {
    "FatMexPromptBatch": "Fat Mex Prompt Batch",
    "FatMexPromptTemplate": "Fat Mex Prompt Template",
}
