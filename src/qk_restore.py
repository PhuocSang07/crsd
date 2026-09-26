"""QK-Restore (D6): a copy of a trained adapter whose q_proj/k_proj update is zeroed.

    python src/qk_restore.py --adapter checkpoints/csrd-q8b-1.7b --output-dir checkpoints/csrd-q8b-1.7b-qkrestore

The copy routes with the pre-training W_Q, W_K and keeps every other learned update; evaluate it
like any adapter, and extract its routing with extract_routing.py --qk-restore. Reading (Sec. 5):
if SFT barely changes under QK-Restore while CSRD's gain over SFT shrinks by at least half,
the gain is attributable to the routing parameters (Hypothesis 4(iii)).
For a CSRD-QK run pass <ckpt>/adapters-separate/default (the Q/K adapter is dropped entirely).
"""

import argparse

from model_utils import write_qk_restored_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    write_qk_restored_adapter(args.adapter, args.output_dir)
    print(f"QK-restored adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
