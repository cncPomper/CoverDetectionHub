import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)
from evaluation.covers80_eval import evaluate_on_covers80
from evaluation.abracadabra_eval import evaluate_on_injected_abracadabra
from evaluation.datacos_eval import evaluate_on_datacos
from evaluation.distracted_dataset_eval import evaluate_on_distracted_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cover song detection models on datasets.")

    # Add argument for model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["ByteCover", "CoverHunter", "Lyricover", "Lyricover Augmented", "MFCC", "Spectral Centroid", "Remove"],
        help="The name of the model to evaluate (ByteCover, CoverHunter, Lyricover, Lyricover Augmented, MFCC, Spectral Centroid, Remove)."
    )

    # Add argument for dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["covers80", "covers80but10", "injected_abracadabra","da_tacos", "distracted_dataset", "distracted_dataset_reference"],
        help="The dataset to evaluate on (covers80, covers80but10, injected_abracadabra, da_tacos, distracted_dataset, distracted_dataset_reference)."
    )

    # Add optional argument for precision-at-k
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="The value of k for precision-at-k (default: 10)."
    )

    args = parser.parse_args()

    # Determine the dataset and evaluate
    if args.dataset in ["covers80", "covers80but10"]:
        covers80but10 = args.dataset == "covers80but10"
        result = evaluate_on_covers80(
            model_name=args.model,
            covers80but10=covers80but10,
            k=args.k
        )
    elif args.dataset == "injected_abracadabra":
        result = evaluate_on_injected_abracadabra(
            model_name=args.model,
            k=args.k
        )
    elif args.dataset == "da_tacos":
        result = evaluate_on_datacos(
            model_name=args.model,
            k=args.k
        )
    elif args.dataset == ["distracted_dataset","distracted_dataset_reference"]:
        reference = args.dataset == "distracted_dataset_reference"
        result = evaluate_on_distracted_dataset(
            model_name=args.model,
            reference=reference,
            k=args.k
        )
    else:
        raise ValueError("Unsupported dataset. Choose from covers80, covers80but10, injected_abracadabra, da_tacos, distracted_dataset, distracted_dataset_reference")

    # Print the results
    print("Evaluation Results:")
    for key, value in result.items():
        print(f"{key}: {value}")
