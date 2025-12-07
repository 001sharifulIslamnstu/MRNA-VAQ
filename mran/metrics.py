from typing import Dict, List

import torch
from torch import Tensor

try:
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
except Exception:  # if nltk not fully installed
    corpus_bleu = None
    meteor_score = None


def compute_classification_accuracy(logits: Tensor, labels: Tensor) -> float:
    """
    Simple top-1 accuracy for classification.
    """
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).float().sum().item()
    total = labels.numel()
    return correct / max(total, 1)


def compute_text_metrics(
    pred_answers: List[str],
    ref_answers: List[List[str]],
) -> Dict[str, float]:
    """
    Compute BLEU and METEOR from answer strings.

    pred_answers: [N]
    ref_answers: [N][K]  (list of lists of reference answers)
    """
    metrics = {}

    # BLEU
    if corpus_bleu is not None:
        refs_bleu = [[r.split() for r in refs] for refs in ref_answers]
        hyps_bleu = [p.split() for p in pred_answers]
        bleu = corpus_bleu(refs_bleu, hyps_bleu)
        metrics["BLEU"] = bleu
    else:
        metrics["BLEU"] = -1.0

    # METEOR
    if meteor_score is not None:
        meteors = []
        for pred, refs in zip(pred_answers, ref_answers):
            meteors.append(meteor_score(refs, pred))
        metrics["METEOR"] = float(sum(meteors) / max(len(meteors), 1))
    else:
        metrics["METEOR"] = -1.0

    return metrics
