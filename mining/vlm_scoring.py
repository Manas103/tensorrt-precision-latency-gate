"""Scores every frame with a real vision-language model: CLIP (openai/clip-vit-base-patch32),
zero-shot against 4 text prompts, one per scenario class. CLIP is a vision-language model in the
literal sense the term is used in the literature (an image encoder and a text encoder trained
jointly, contrastively, on image-caption pairs); it is not a generative VLM like LLaVA, and this
README says so rather than implying otherwise. Scoring is image-text cosine similarity, softmax'd
over the 4 prompts per frame; the argmax is the VLM's predicted scenario label for that frame.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from mining.dataset import CLASSES

MODEL_NAME = "openai/clip-vit-base-patch32"

def _as_feature_tensor(output) -> torch.Tensor:
    """transformers 5.x's CLIPModel.get_text_features / get_image_features return a
    BaseModelOutputWithPooling whose `pooler_output` is the projected embedding, not a bare
    tensor; older versions returned the tensor directly. Handle both."""
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


PROMPTS = {
    "normal": "a photo of a dry, matte gray asphalt road under flat daylight",
    "low_light": "a photo of a dark, poorly lit road at night",
    "wet_surface": "a photo of a wet road with glossy reflections and a bluish sheen",
    "construction_zone": "a photo of a road construction zone with orange cones and barriers",
}


class ClipSceneScorer:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        prompts_in_class_order = [PROMPTS[c] for c in CLASSES]
        with torch.no_grad():
            text_inputs = self.processor(text=prompts_in_class_order, return_tensors="pt", padding=True)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_features = _as_feature_tensor(self.model.get_text_features(**text_inputs))
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score_batch(self, images_hwc_uint8: np.ndarray) -> np.ndarray:
        """images_hwc_uint8: (N, 32, 32, 3) uint8. Returns (N, len(CLASSES)) softmax probabilities."""
        pixel_values = self.processor(images=list(images_hwc_uint8), return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)
        image_features = _as_feature_tensor(self.model.get_image_features(pixel_values=pixel_values))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = 100.0 * image_features @ self.text_features.T
        probs = logits.softmax(dim=-1)
        return probs.cpu().numpy()

    def predict_labels(self, images_hwc_uint8: np.ndarray) -> list[str]:
        probs = self.score_batch(images_hwc_uint8)
        return [CLASSES[i] for i in probs.argmax(axis=1)]
