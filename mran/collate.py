from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision import transforms
from transformers import AutoTokenizer


class MRANCollator:
    """
    Collator: image preprocessing + question tokenization.

    - Uses ViT-style image transforms (224x224, ImageNet normalization).
    - Uses HuggingFace tokenizer for questions.
    """

    def __init__(
        self,
        text_encoder_name: str,
        max_question_len: int = 32,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.max_question_len = max_question_len

        self.img_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        images = [self.img_transform(x["image"]) for x in batch]
        questions = [x["question"] for x in batch]
        answer_ids = [x["answer_id"] for x in batch]
        masks = [x["mask"] for x in batch]

        pixel_values = torch.stack(images, dim=0)  # (B, 3, 224, 224)
        answer_ids = torch.tensor(answer_ids, dtype=torch.long)

        text_tokens = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_question_len,
            return_tensors="pt",
        )

        # masks -> AGS masks (B, P) or None
        ags_masks: Optional[Tensor] = None
        if any(m is not None for m in masks):
            valid_masks: List[Tensor] = []
            for m in masks:
                if m is None:
                    valid_masks.append(torch.zeros(1))  # dummy, will be broadcast later
                else:
                    valid_masks.append(torch.tensor(m, dtype=torch.float))
            ags_masks = torch.stack(valid_masks, dim=0)

        return {
            "pixel_values": pixel_values,
            "input_ids": text_tokens["input_ids"],
            "attention_mask": text_tokens["attention_mask"],
            "answer_ids": answer_ids,
            "ags_masks": ags_masks,
        }
