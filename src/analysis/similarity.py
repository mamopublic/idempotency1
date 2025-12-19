import difflib
import os
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np

class EmbeddingModel:
    def encode_text(self, text):
        raise NotImplementedError
    def encode_image(self, image_path):
        raise NotImplementedError
    def compute_similarity(self, emb1, emb2):
        raise NotImplementedError

class SigLIPEmbedding(EmbeddingModel):
    def __init__(self, model_name="google/siglip-so400m-patch14-384"):
        from transformers import AutoProcessor, SiglipModel
        import torch
        self.device = "cpu"
        self.model = SiglipModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.torch = torch

    def encode_text(self, text):
        # SigLIP expects specific padding/truncation
        inputs = self.processor(text=[text], return_tensors="pt", padding="max_length", truncation=True).to(self.device)
        with self.torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
        # Normalize
        return outputs / outputs.norm(dim=-1, keepdim=True)

    def encode_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
        # Normalize
        return outputs / outputs.norm(dim=-1, keepdim=True)

    def compute_similarity(self, emb1, emb2):
        # Dot product of normalized vectors = Cosine Similarity
        return (emb1 @ emb2.T).item()

class TitanEmbedding(EmbeddingModel):
    def __init__(self, model_name="amazon.titan-embed-image-v1"):
        import boto3
        import json
        self.client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.model_id = model_name

    def _invoke(self, body):
        import json
        response = self.client.invoke_model(
            body=json.dumps(body),
            modelId=self.model_id,
            accept="application/json",
            contentType="application/json"
        )
        return json.loads(response.get("body").read())

    def encode_text(self, text):
        body = {"inputText": text}
        response = self._invoke(body)
        return np.array(response["embedding"])

    def encode_image(self, image_path):
        import base64
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        body = {"inputImage": image_data}
        response = self._invoke(body)
        return np.array(response["embedding"])

    def compute_similarity(self, emb1, emb2):
        # Cosine similarity for numpy arrays
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(emb1, emb2) / (norm1 * norm2)

class SimilarityEngine:
    def __init__(self, config):
        model_type = config["models"].get("embedding_type", "siglip") # Default to siglip
        model_name = config["models"]["embedding_model"]
        
        if model_type == "titan":
            self.model = TitanEmbedding(model_name)
        else:
            self.model = SigLIPEmbedding(model_name)

    def compute_text_similarity(self, text1, text2):
        """Computes character-level similarity (0.0 to 1.0)."""
        return difflib.SequenceMatcher(None, text1, text2).ratio()

    def compute_semantic_similarity(self, text1, text2):
        """Computes semantic similarity between two texts."""
        emb1 = self.model.encode_text(text1)
        emb2 = self.model.encode_text(text2)
        return self.model.compute_similarity(emb1, emb2)

    def compute_cross_modal_similarity(self, text, image_path):
        """Computes similarity between text and image."""
        emb_text = self.model.encode_text(text)
        emb_image = self.model.encode_image(image_path)
        return self.model.compute_similarity(emb_text, emb_image)

    def compute_visual_similarity(self, image_path1, image_path2):
        """Computes similarity between two images."""
        emb1 = self.model.encode_image(image_path1)
        emb2 = self.model.encode_image(image_path2)
        return self.model.compute_similarity(emb1, emb2)
