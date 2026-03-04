import argparse

from circuitlab.clt_training_runner import CLTTrainingRunner
from runners.decoder_comparison.config import clt_training_runner_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder_type", type=str, default="full",
                        choices=["full", "lora", "linear_transform"])
    parser.add_argument("--decoder_rank", type=int, default=64)
    args = parser.parse_args()

    cfg = clt_training_runner_config(
        decoder_type=args.decoder_type,
        decoder_rank=args.decoder_rank,
    )
    runner = CLTTrainingRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
