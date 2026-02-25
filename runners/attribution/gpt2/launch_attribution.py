import torch
from circuitlab.attribution.attribution import AttributionRunner
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# should run on a single GPU

def main():
    clt_checkpoint = "/fast/fdraye/data/featflow/cache/checkpoints/gpt2/d1s3fw30/middle_22137856"
    model_name = "gpt2"

    test_strings = [
        'The opposite of "large" is "'
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # where to save the attribution_graph.pt
    folder_name = "/home/fdraye/projects/circuitlab/save"

    print("Creating Attribution Runner")
    runner = AttributionRunner(
        clt_checkpoint=clt_checkpoint,
        model_name=model_name,
        device=device,
        debug=True,
    )

    for i, test_string in enumerate(test_strings, 1):
        print(f"\n Processing test string {i}: '{test_string}'")

        try:
            _ = runner.run(
                input_string=test_string,
                folder_name=folder_name,
                max_n_logits=1,
                desired_logit_prob=0.95,
                max_feature_nodes=1_000,
                batch_size=256,
                feature_threshold=0.80,
                edge_threshold=0.98,
                offload=None,
                run_interventions=True,
            )

        except Exception as e:
            print(f"Attribution computation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
