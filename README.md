# MRAN-VQA: Multimodal Recursive Attention Network for Visual Question Answering

This repository contains a PyTorch implementation of **MRAN-VQA** and the **Attention Grounding Score (AGS)** following:

> MRAN-VQA: Multimodal Recursive Attention Network for Visual Question Answering  
> *Engineering Science and Technology, an International Journal (JESTECH), 2025, Vol. 72, Article 102232*  
> DOI: **10.1016/j.jestch.2025.102232**

The code supports:

- Training **MRAN-VQA** on **VQA v2.0**, **CLEVR**, and **BanglaVQA**
- Computing **VQA accuracy**, **BLEU**, **METEOR**, and **AGS+**
- **Recursive attention depth sweep** (R = 1–6) with latency measurement
- **Ablation studies** (w/o Recursive Attention, w/o Hierarchical Fusion, w/o AGS loss)
- **Model statistics** (parameter count, single-sample latency)
- Simple scripts for **qualitative examples** and **AGS analysis**

---

## 1. Repository Structure

```text
.
├── main.py                       # Entry point: train / sweep_R / ablation / stats
├── mran/
│   ├── __init__.py
│   ├── config.py                 # TrainingConfig dataclass
│   ├── datasets.py               # Generic VQA-style dataset loader
│   ├── collate.py                # Collator: image transforms + tokenization
│   ├── ags.py                    # Attention Grounding Score (AGS) + loss
│   ├── model.py                  # MRAN-VQA model + recursive attention block
│   ├── metrics.py                # VQA soft accuracy, BLEU, METEOR, helpers
│   ├── train_eval.py             # Training loop + evaluation
│   ├── recursion_sweep.py        # R-sweep experiments (accuracy + latency)
│   ├── ablations.py              # Ablation study utilities
│   ├── model_stats.py            # Parameter count + latency measurement
│   └── qualitative.py            # Qualitative examples / visualization helper
└── README.md                     # This file
```

---

## 2. Installation

> Python **>= 3.9** recommended.

```bash
# Core libraries
pip install torch torchvision torchaudio  # add CUDA index-url if needed

# Transformers, metrics, utilities
pip install transformers==4.45.0
pip install pillow tqdm nltk

# Optional: for BLEU/METEOR
python -m nltk.downloader punkt wordnet omw-1.4
```

---

## 3. Data Preparation

The training code expects **preprocessed JSON/JSONL** files with entries of the form:

```jsonc
{
  "image": "COCO_train2014_000000123456.jpg",  // relative to image_root
  "question": "What color is the car?",
  "answers": ["red", "red", "red", "blue", ...],  // 10 VQA-style annotations
  "answer": "red",                  // majority answer (string)
  "answer_id": 123,                 // index in answer vocabulary
  "mask": [0, 1, 0, ..., 0]         // (optional) AGS mask over ViT patches (len = num_patches)
}
```

**Note**

- You can create these JSON files using your own preprocessing scripts for **VQA v2.0**, **CLEVR**, and **BanglaVQA**.
- The code includes helpers to **build an answer vocabulary** and **attach `answer_id`** fields.

### 3.1. Answer Vocabulary

This is automatically called inside `train_mran()` if the vocab file does not exist, but you can also run it manually:

```python
from mran.train_eval import build_answer_vocab

build_answer_vocab(
    train_json="data/vqa_v2/train.json",
    vocab_path="data/vqa_v2/answers_vocab.json",
    top_k=3000
)
```

This creates a JSON file with `ans2id` and `id2ans` mappings.

### 3.2. Attaching `answer_id`

If your JSON does not yet contain an `answer_id` field:

```python
from mran.train_eval import attach_answer_ids, load_answer_vocab

ans2id, id2ans = load_answer_vocab("data/vqa_v2/answers_vocab.json")

attach_answer_ids(
    json_in="data/vqa_v2/train_raw.json",
    json_out="data/vqa_v2/train.json",
    ans2id=ans2id
)

attach_answer_ids(
    json_in="data/vqa_v2/val_raw.json",
    json_out="data/vqa_v2/val.json",
    ans2id=ans2id
)
```

---

## 4. Training MRAN-VQA

All main experiments are launched through `main.py`.

### 4.1. General Command

```bash
python main.py   --mode train   --dataset <vqa_v2|clevr|bangla_vqa>   --train_json /path/to/train.json   --val_json /path/to/val.json   --image_root /path/to/images   --answer_vocab /path/to/answers_vocab.json   --recursion_depth 4   --device cuda
```

### 4.2. VQA v2.0

```bash
python main.py   --mode train   --dataset vqa_v2   --train_json data/vqa_v2/train.json   --val_json data/vqa_v2/val.json   --image_root /data/coco/train2014   --answer_vocab data/vqa_v2/answers_vocab.json   --recursion_depth 4   --device cuda
```

Configuration:

- Encoder: **ViT-base** + **bert-base-uncased**
- Epochs: **40**
- Metrics: VQA soft accuracy, BLEU, METEOR, **AGS+** (if masks are provided)

### 4.3. CLEVR

```bash
python main.py   --mode train   --dataset clevr   --train_json data/clevr/train.json   --val_json data/clevr/val.json   --image_root /data/clevr/images   --answer_vocab data/clevr/answers_vocab.json   --recursion_depth 4   --device cuda
```

- Epochs: **20** (default in config for CLEVR)
- If AGS masks are available for CLEVR (e.g., object or region labels), include `"mask"` in the JSON entries to train with AGS loss.

### 4.4. BanglaVQA

```bash
python main.py   --mode train   --dataset bangla_vqa   --train_json data/bangla_vqa/train.json   --val_json data/bangla_vqa/val.json   --image_root /data/bangla_vqa/images   --answer_vocab data/bangla_vqa/answers_vocab.json   --recursion_depth 4   --device cuda
```

- Uses multilingual BERT: **bert-base-multilingual-cased** for the question encoder.
- Epochs: **30** (default for BanglaVQA).
- Reports: VQA accuracy, BLEU, METEOR, and AGS+ (if masks are provided).

Trained checkpoints are saved as:

```text
mran_<dataset>_best.pt
# e.g., mran_vqa_v2_best.pt
```

---

## 5. Recursive Depth Sweep (R = 1–6)

To reproduce the recursive attention depth vs. accuracy and latency experiments:

```bash
python main.py   --mode sweep_R   --dataset vqa_v2   --train_json data/vqa_v2/train.json   --val_json data/vqa_v2/val.json   --image_root /data/coco/train2014   --answer_vocab data/vqa_v2/answers_vocab.json   --device cuda
```

This will:

1. Load the data.
2. For each recursion depth `R` in `{1, 2, 3, 4, 5, 6}`:
   - Evaluate the model.
   - Measure **per-sample latency (ms)** over several batches.
3. Print a list of dictionaries like:

```text
{
  "R": 4,
  "metrics": {
    "accuracy": 75.6,
    "BLEU": ...,
    "METEOR": ...,
    "AGS_plus": ...
  },
  "latency_ms": 12.3
}
```

You can use these results to construct the **R vs. accuracy vs. latency** plots in the paper.

> For best fidelity, load a previously trained checkpoint into the base model in `recursion_sweep.py` (the loading line is already commented where you need it).

---

## 6. Ablation Studies

To run the ablations (without Recursive Attention, Hierarchical Fusion, or AGS loss):

```bash
python main.py   --mode ablation   --dataset vqa_v2   --train_json data/vqa_v2/train.json   --val_json data/vqa_v2/val.json   --image_root /data/coco/train2014   --answer_vocab data/vqa_v2/answers_vocab.json   --device cuda
```

The script will:

1. Construct a **base MRAN-VQA** model.
2. Evaluate the following variants:
   - Full **MRAN-VQA**
   - **w/o RA** (no recursive attention)
   - **w/o HF** (no hierarchical fusion)
   - **w/o AGS** (no AGS auxiliary loss)
3. Print metrics for each variant.

> For full fidelity with the paper, load your trained checkpoint into `base_model` in `ablations.py` (the loading line is already provided as a comment).

---

## 7. Model Statistics & Latency

To approximate **parameter counts** and **single-sample latency** (for table-style comparisons):

```bash
python main.py --mode stats --dataset vqa_v2   --train_json dummy.json   --val_json dummy.json   --image_root .   --answer_vocab data/vqa_v2/answers_vocab.json
```

> Only `--device` really matters here; the JSON paths are **not** used in `stats` mode.

Internally, `print_mran_stats()`:

- Instantiates an MRAN-VQA with ViT + BERT.
- Prints total **trainable parameters**.
- Measures single-sample inference **latency (ms)** for batch size 1.

You can adapt `model_stats.py` to compare with other architectures (e.g., MCAN, BAN, BUTD, ViLT).

---

## 8. Qualitative Examples & AGS Visualization

You can load a trained MRAN-VQA checkpoint and run qualitative examples:

```python
from mran.qualitative import run_qualitative_examples

run_qualitative_examples(
    model_ckpt="mran_vqa_v2_best.pt",
    image_paths=[
        "examples/img1.jpg",
        "examples/img2.jpg",
    ],
    questions=[
        "What is the man holding?",
        "How many dogs are there?",
    ],
    id2ans=None,        # id2ans is loaded from the checkpoint
    device_str="cuda"
)
```

This prints predicted answers for the provided image–question pairs.  
You can also access the attention maps in `model.forward()` to visualize **AGS-aligned regions**.

---

## 9. Using AGS with Other Models

The AGS implementation is **model-agnostic** and can be used with any attention map over patches or regions:

```python
import torch
from mran.ags import compute_ags

# a: (B, P) attention over P patches/regions (sum to 1 per sample)
# g: (B, P) binary ground-truth mask (1 = relevant, 0 = irrelevant)
a = torch.rand(8, 196)
a = a / a.sum(dim=1, keepdim=True)
g = torch.randint(0, 2, (8, 196)).float()

ags_raw, ags_plus = compute_ags(a, g, lam=0.5)
print("AGS:", ags_raw.mean().item())
print("AGS+:", ags_plus.mean().item())
```

You can plug in attention from:

- **ViLT, MCAN, BAN, BUTD**
- **LLM-based VQA models** with visual attention
- Any custom architecture that produces a distribution over image tokens

---

## 10. Configuration

Most hyper-parameters are controlled via `TrainingConfig` in `mran/config.py`, including:

- `image_encoder_name` (default: `google/vit-base-patch16-224-in21k`)
- `text_encoder_name` (default: `bert-base-uncased`)
- `text_encoder_name_multilingual` (for BanglaVQA, default: `bert-base-multilingual-cased`)
- `hidden_dim` (fusion dimension, default: `768`)
- `recursion_depth` (`R`, default: `4`)
- Training parameters:
  - `batch_size`
  - `epochs`
  - `lr`
  - `weight_decay`
  - `warmup_epochs`
- Loss flags:
  - `use_recursive_attention`
  - `use_hierarchical_fusion`
  - `use_ags_loss`
  - `gamma_ags`

You can:

- Modify defaults in `config.py`, or  
- Override them by editing `make_cfg()` in `main.py`.

---

## 11. Citing MRAN-VQA

If you use this code, the **MRAN-VQA model**, or the **BanglaVQA dataset** in your research, please cite:

### Plain Text Citation

> M. S. Islam, M. A. T. Rony, M. M. H. Sarker, M. K. B. Bhuiyan, M. Saib, M. Aktarujjaman,  
> M. S. Uddin, A. D. Algarni, A. T. Azar, and W. El-Shafai,  
> “MRAN-VQA: Multimodal Recursive Attention Network for Visual Question Answering,”  
> *Engineering Science and Technology, an International Journal*, vol. 72, art. no. 102232, 2025.  
> doi: **10.1016/j.jestch.2025.102232**

### BibTeX

```bibtex
@article{Islam2025MRANVQA,
  title   = {MRAN-{VQA}: Multimodal Recursive Attention Network for Visual Question Answering},
  author  = {Islam, Mohammad Shariful and Rony, Mohammad Abu Tareq and Sarker, Md Murad Hossain
             and Bhuiyan, Md Khairul Bashar and Saib, Md and Aktarujjaman, Md and Uddin, Md Shahab
             and Algarni, Abeer D. and Azar, Ahmad Taher and El-Shafai, Walid},
  journal = {Engineering Science and Technology, an International Journal},
  volume  = {72},
  pages   = {102232},
  year    = {2025},
  issn    = {2215-0986},
  doi     = {10.1016/j.jestch.2025.102232},
  note    = {Open access under CC BY-NC-ND 4.0},
  publisher = {Elsevier}
}
```

> If you update the author list formatting or add extra BibTeX fields (e.g., `url`), please follow the official journal record.

---

## 12. License

- The **paper** is published under **CC BY-NC-ND 4.0** (see the journal page for details).
- The **code** in this repository can be licensed as you prefer (e.g., **MIT**, **Apache 2.0**, or a custom research license).

Create a separate `LICENSE` file (e.g., `MIT` or `Apache-2.0` template) and update this section with the chosen license.

---

## 13. Contact

For questions about:

- The **MRAN-VQA model** or this implementation:  
  Please contact the corresponding author (as listed in the paper) or open an issue in your repository.

For questions about:

- The **BanglaVQA dataset** or additional resources:  
  Please refer to the dataset section in the paper or contact the authors directly.

---

Happy experimenting! 🎯🧠📷
