"""
Disease Predictor Module
========================
Provides the `DiseasePredictor` inference engine that loads trained model weights,
runs image preprocessing, executes neural network forward pass, and enriches predictions
with structured disease metadata, symptoms, causes, and treatment remedies.
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from src.disease.class_mapping import (
        PD_CLASS_TO_IDX,
        PD_IDX_TO_CLASS,
        NUM_PD_CLASSES,
        parse_class,
    )
    from src.disease.disease_info import get_disease_info
    from src.disease.dataset import get_transforms
    from src.disease.model import build_model
except ImportError:
    from class_mapping import (
        PD_CLASS_TO_IDX,
        PD_IDX_TO_CLASS,
        NUM_PD_CLASSES,
        parse_class,
    )
    from disease_info import get_disease_info
    from dataset import get_transforms
    from model import build_model


class DiseasePredictor:
    """
    Inference wrapper for plant leaf disease classification.
    """

    def __init__(
        self,
        model_path: str = "weights/leaf_disease_model_final.pth",
        backbone: str = "mobilenet_v3_large",
        device: Optional[str] = None,
        img_size: int = 224,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_path = model_path
        self.backbone = backbone
        self.img_size = img_size
        self.class_to_idx = PD_CLASS_TO_IDX.copy()
        self.idx_to_class = PD_IDX_TO_CLASS.copy()
        self.num_classes = NUM_PD_CLASSES

        self.model = self._load_model()
        self.transform = get_transforms(is_train=False, img_size=img_size)

    def _load_model(self) -> torch.nn.Module:
        """Loads model architecture and checkpoint weights."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at '{self.model_path}'.\n"
                f"Please run `train_plantdoc.py` to train and save the fine-tuned model first."
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Restore class mappings if stored in checkpoint
        if isinstance(checkpoint, dict):
            if "class_to_idx" in checkpoint:
                self.class_to_idx = checkpoint["class_to_idx"]
                self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            if "idx_to_class" in checkpoint:
                self.idx_to_class = {
                    int(k): v for k, v in checkpoint["idx_to_class"].items()
                }
            if "num_classes" in checkpoint:
                self.num_classes = checkpoint["num_classes"]
            elif "class_to_idx" in checkpoint:
                self.num_classes = len(self.class_to_idx)
            if "backbone" in checkpoint:
                self.backbone = checkpoint["backbone"]

            state_dict = checkpoint.get("model_state_dict", checkpoint)
        else:
            state_dict = checkpoint

        model = build_model(
            num_classes=self.num_classes,
            backbone=self.backbone,
            pretrained_imagenet=False,
        )
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def predict(self, image_input: Union[str, np.ndarray, Image.Image], top_k: int = 3) -> Dict[str, Any]:
        """
        Runs disease prediction on an image file path, numpy RGB array, or PIL Image.

        Args:
            image_input: File path (str), NumPy array (RGB or BGR), or PIL Image.
            top_k: Number of highest-confidence classes to return.

        Returns:
            Dictionary containing structured prediction results.
        """
        # Convert input to RGB numpy array
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Input image file not found: {image_input}")
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise ValueError(f"Could not decode image at {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                img_rgb = image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Preprocess through albumentations transforms
        augmented = self.transform(image=img_rgb)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1).squeeze(0)

        # Get top-k predictions
        top_k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probabilities, k=top_k)

        top_predictions: List[Dict[str, Any]] = []
        for prob, idx_t in zip(top_probs, top_indices):
            idx = idx_t.item()
            cls_name = self.idx_to_class.get(idx, f"Unknown Class {idx}")
            crop, is_healthy, disease = parse_class(cls_name)
            top_predictions.append(
                {
                    "class_index": idx,
                    "class_name": cls_name,
                    "crop": crop,
                    "is_healthy": is_healthy,
                    "disease": disease,
                    "confidence": round(prob.item() * 100.0, 2),
                    "probability": float(prob.item()),
                }
            )

        # Top 1 details
        best = top_predictions[0]
        disease_info = get_disease_info(best["class_name"])

        return {
            "predicted_class": best["class_name"],
            "confidence": best["confidence"],
            "crop": best["crop"],
            "is_healthy": best["is_healthy"],
            "status": "Healthy" if best["is_healthy"] else "Diseased",
            "disease": best["disease"],
            "top_k": top_predictions,
            "disease_info": disease_info,
        }

    def predict_from_array(self, image_array: np.ndarray, top_k: int = 3) -> Dict[str, Any]:
        """Convenience method for predicting from in-memory RGB image arrays."""
        return self.predict(image_array, top_k=top_k)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python disease_predictor.py <path_to_leaf_image>")
        print("Example: python disease_predictor.py PlantDoc-Dataset/test/Tomato\\ leaf\\ late\\ blight/sample.jpg")
        sys.exit(0)

    test_img = sys.argv[1]
    default_model = "weights/leaf_disease_model_final.pth"

    if not os.path.exists(default_model):
        print(f" Model checkpoint '{default_model}' not found.")
        print("Please train the model first using:")
        print("  python train_plantvillage.py --epochs 15")
        print("  python train_plantdoc.py --stage1 weights/plantvillage_pretrained.pth --epochs 25")
        sys.exit(1)

    predictor = DiseasePredictor(model_path=default_model)
    result = predictor.predict(test_img, top_k=3)
    print("\n--- Prediction Result ---")
    print(f"Class:      {result['predicted_class']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Crop:       {result['crop']}")
    print(f"Status:     {result['status']}")
    print(f"Disease:    {result['disease']}")
