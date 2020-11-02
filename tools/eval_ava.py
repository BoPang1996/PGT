import os
import sys
import argparse

import slowfast.utils.logging as logging
from slowfast.utils.parser import load_config
from slowfast.utils.ava_eval_helper import evaluate_ava_from_files


def parse_args():
    parser = argparse.ArgumentParser(description="AVA evaluator.")
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args)

    logging.setup_logger(cfg, 'test')
    evaluate_ava_from_files(
        os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.LABEL_MAP_FILE),
        os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.GROUNDTRUTH_FILE),
        os.path.join(cfg.LOGS.DIR, "detections_latest.csv"),
        os.path.join(cfg.AVA.ANNOTATION_DIR, cfg.AVA.EXCLUSION_FILE)
    )


if __name__ == "__main__":
    main()
