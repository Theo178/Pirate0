"""
Feature Extraction - Visual embedding generation with augmentation
Uses pretrained ResNet50 with multi-variant reference embeddings
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image
from typing import List


# Augmentation variants applied to reference frames at ingestion time.
# Each variant simulates a common piracy transformation so the index
# can match cropped, flipped, zoomed, or rotated copies.
AUGMENTATION_LABELS = [
    'original',
    'crop_80',       # 80% center crop -- simulates watermark/letterbox removal
    'horizontal_flip',  # Mirror -- common evasion technique
    'rotate_5',      # +5 degree rotation -- cam-rip misalignment
    'zoom_110',      # 110% zoom center crop -- slight scale change
]


class FeatureExtractor:
    """Extract visual features from frames using ResNet50"""

    def __init__(self, device: str = None):
        """
        Initialize feature extractor.

        Args:
            device: 'cuda' or 'cpu' (auto-detects if None)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove classification head
        self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Disable gradients
        for param in self.model.parameters():
            param.requires_grad = False

        # Standard preprocessing (applied after any augmentation)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.embedding_dim = 2048

    def _apply_augmentations(self, img: Image.Image) -> List[Image.Image]:
        """
        Generate augmented variants of an image for reference indexing.

        Args:
            img: PIL Image (RGB)

        Returns:
            List of PIL Images: [original, crop_80, h_flip, rotate_5, zoom_110]
        """
        w, h = img.size
        variants = [img]  # original

        # 1. Center crop at 80% -- simulates watermark/border removal
        crop_w, crop_h = int(w * 0.8), int(h * 0.8)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        variants.append(img.crop((left, top, left + crop_w, top + crop_h)))

        # 2. Horizontal flip -- mirror
        variants.append(TF.hflip(img))

        # 3. Slight rotation (+5 degrees) -- cam-rip misalignment
        variants.append(TF.rotate(img, angle=5, expand=False, fill=0))

        # 4. Zoom 110% via center crop then resize back -- scale change
        zoom_crop_w, zoom_crop_h = int(w * 0.9), int(h * 0.9)
        z_left = (w - zoom_crop_w) // 2
        z_top = (h - zoom_crop_h) // 2
        zoomed = img.crop((z_left, z_top, z_left + zoom_crop_w, z_top + zoom_crop_h))
        zoomed = zoomed.resize((w, h), Image.LANCZOS)
        variants.append(zoomed)

        return variants

    def extract_from_path(self, image_path: str) -> np.ndarray:
        """
        Extract features from image file (single embedding, no augmentation).

        Args:
            image_path: Path to image file

        Returns:
            Feature vector (2048-dim)
        """
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(img_tensor)

        return features.cpu().numpy().flatten()

    def extract_augmented(self, image_path: str) -> np.ndarray:
        """
        Extract features for all augmented variants of an image.
        Used during reference ingestion to build a robust index.

        Args:
            image_path: Path to image file

        Returns:
            Feature matrix (num_augmentations x 2048)
        """
        img = Image.open(image_path).convert('RGB')
        variants = self._apply_augmentations(img)

        # Batch all variants through the model at once
        tensors = torch.stack([self.transform(v) for v in variants]).to(self.device)

        with torch.no_grad():
            features = self.model(tensors)

        return features.cpu().numpy().reshape(len(variants), -1)

    def extract_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        Extract features from multiple images (single embedding each).
        Used for query frames.

        Args:
            image_paths: List of image paths

        Returns:
            Feature matrix (N x 2048)
        """
        embeddings = []
        for path in image_paths:
            emb = self.extract_from_path(path)
            embeddings.append(emb)

        return np.stack(embeddings, axis=0)

    def extract_batch_augmented(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Extract augmented features for multiple images.
        Used for reference frame ingestion.

        Args:
            image_paths: List of image paths

        Returns:
            List of feature matrices, one per image.
            Each matrix is (num_augmentations x 2048).
        """
        results = []
        for path in image_paths:
            aug_embeddings = self.extract_augmented(path)
            results.append(aug_embeddings)
        return results
