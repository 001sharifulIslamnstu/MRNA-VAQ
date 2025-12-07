import argparse
from pathlib import Path

from mran.config import get_default_config, TrainingConfig
from mran.train_eval import train_mran, print_mran_stats
from mran.recursion_sweep import run_recursion_sweep
from mran.ablations import run_ablations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MRAN-VQA: Multimodal Recursive Attention Network for VQA"
    )
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "sweep_R", "ablation", "stats"],
                        help="Experiment mode")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["vqa_v2", "clevr", "bangla_vqa"])
    parser.add_argument("--train_json", type=str, required=False, default=None)
    parser.add_argument("--val_json", type=str, required=False, default=None)
    parser.add_argument("--image_root", type=str, required=False, default=None)
    parser.add_argument("--answer_vocab", type=str, required=False, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--recursion_depth", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional checkpoint path for sweep/ablations")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> TrainingConfig:
    cfg = get_default_config(args.dataset)

    if args.train_json:
        cfg.train_json = args.train_json
    if args.val_json:
        cfg.val_json = args.val_json
    if args.image_root:
        cfg.image_root = args.image_root
    if args.answer_vocab:
        cfg.answer_vocab = args.answer_vocab

    cfg.output_dir = args.output_dir
    cfg.recursion_depth = args.recursion_depth
    cfg.device = args.device

    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)

    if args.mode == "train":
        train_mran(cfg)
    elif args.mode == "sweep_R":
        run_recursion_sweep(cfg, ckpt_path=args.ckpt)
    elif args.mode == "ablation":
        run_ablations(cfg, ckpt_path=args.ckpt)
    elif args.mode == "stats":
        print_mran_stats(cfg)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
