import os
import pandas as pd
import gradio as gr
from tqdm import tqdm
import yaml
import logging
from csi_models.ModelBase import ModelBase
from csi_models.ByteCoverModel import ByteCoverModel
from csi_models.CoverHunterModel import CoverHunterModel
from csi_models.LyricoverModel import LyricoverModel
from csi_models.LyricoverAugmentedModel import LyricoverAugmentedModel
from csi_models.RemoveModel import RemoveModel
from feature_extraction.feature_extraction import MFCCModel, SpectralCentroidModel
from evaluation.metrics import compute_mean_metrics_for_rankings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration
logging.info("Loading configurations from 'configs/paths.yaml'")
with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)

INJECTED_ABRACADABRA_DIR = config["injected_abracadabra_dir"]
INJECTED_ABRACADABRA_DATA_DIR = config["injected_abracadabra_data_dir"]
GROUND_TRUTH_FILE = config["injected_abracadabra_ground_truth_file"]
REFERENCE_SONG = config["injected_abracadabra_reference_song"]


def gather_injected_abracadabra_files(dataset_path, ground_truth_file):
    """
    Return (file_path, is_injected) pairs by reading from the injection_list CSV.
    """
    logging.info(f"Gathering dataset files from {dataset_path}")
    if not os.path.exists(dataset_path) or not os.path.exists(ground_truth_file):
        logging.error(f"Dataset path or ground truth file does not exist: {dataset_path}, {ground_truth_file}")
        raise FileNotFoundError("Dataset path or ground truth file does not exist.")

    injection_data = pd.read_csv(ground_truth_file)
    logging.info(f"Loaded ground truth data with {len(injection_data)} entries.")

    # Normalize the path by replacing backslashes
    injection_dict = dict(
        zip(
            injection_data['File'].str.replace("\\", "/"),
            injection_data['Injected'].map(lambda x: x == 'Yes')
        )
    )

    all_audio_files = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path) if f.endswith(".wav")
    ]
    logging.info(f"Found {len(all_audio_files)} audio files in the dataset.")

    # Build a list of (path, ground_truth_is_injected)
    files_with_ground_truth = []
    for path in all_audio_files:
        rel_path = os.path.relpath(path, start=INJECTED_ABRACADABRA_DIR).replace("\\", "/")
        is_injected = injection_dict.get(rel_path, False)
        files_with_ground_truth.append((path, is_injected))

    logging.info(f"Matched {len(files_with_ground_truth)} files with ground truth data.")
    return files_with_ground_truth


def compute_ranking_for_reference_song(files_and_labels, model, reference_path=REFERENCE_SONG):
    """
    Compute a ranking of ALL candidate songs by their similarity
    to the ONE given reference.
    """
    logging.info("Starting ranking computation for reference song.")
    if not os.path.exists(reference_path):
        logging.error(f"Reference song file does not exist: {reference_path}")
        raise FileNotFoundError(f"Reference song file does not exist: {reference_path}")

    # Compute reference embedding
    logging.info(f"Computing embedding for reference song: {reference_path}")
    ref_embedding = model.compute_embedding(reference_path)

    # Compute embeddings for all candidate songs
    embeddings = compute_embeddings(files_and_labels, model)

    # Build a list of {candidate_path, similarity, ground_truth} for each candidate
    ranking = []
    for candidate_path, candidate_label in files_and_labels:
        cand_embedding = embeddings[candidate_path]
        sim = model.compute_similarity(ref_embedding, cand_embedding)
        ranking.append({
            "candidate_path": candidate_path,
            "similarity": sim,
            "ground_truth": candidate_label
        })

    # Sort by similarity descending
    ranking.sort(key=lambda x: x["similarity"], reverse=True)

    logging.info("Ranking computation complete.")
    return [
        {
            "query_path": reference_path,
            "query_label": "reference_label",  # or some label you want
            "ranking": ranking  # e.g., list of dicts: {candidate_path, similarity, ground_truth}
        }
    ]


def compute_embeddings(files_and_labels, model: ModelBase, progress=gr.Progress()):
    """
    Compute and cache embeddings for all audio files using the given model.
    """
    logging.info("Starting embedding computation for dataset.")
    progress(0, desc="Calculating embeddings")
    embeddings = {}
    total_files = len(files_and_labels)

    for i, (audio_path, _) in enumerate(tqdm(files_and_labels, desc="Computing embeddings")):
        if audio_path not in embeddings:
            try:
                embeddings[audio_path] = model.compute_embedding(audio_path)
            except Exception as e:
                logging.error(f"Error computing embedding for {audio_path}: {e}")
                continue
        progress((i + 1) / total_files, desc="Calculating embeddings")

    logging.info("Finished computing embeddings.")
    return embeddings


def evaluate_on_injected_abracadabra(model_name, k=8):
    """
    Loads the dataset, computes rankings per song,
    and calculates mAP, mP@k, mMR1.
    """
    logging.info(f"Evaluating model '{model_name}' on injected abracadabra dataset with k={k}.")

    dataset_path = INJECTED_ABRACADABRA_DATA_DIR
    ground_truth_file = GROUND_TRUTH_FILE

    # Select the appropriate model
    model_mapping = {
        "ByteCover": ByteCoverModel,
        "CoverHunter": CoverHunterModel,
        "Lyricover": LyricoverModel,
        "Lyricover Augmented": LyricoverAugmentedModel,
        "MFCC": MFCCModel,
        "Spectral Centroid": SpectralCentroidModel,
        "Remove": RemoveModel
    }

    if model_name not in model_mapping:
        logging.error(f"Unsupported model '{model_name}'.")
        raise ValueError(f"Unsupported model. Choose one of: {list(model_mapping.keys())}")

    model = model_mapping[model_name]()
    logging.info(f"Using model '{model_name}' for evaluation.")

    # Load the list of (file, label)
    files_and_labels = gather_injected_abracadabra_files(dataset_path, ground_truth_file)

    # Compute rankings for the reference song
    ranking = compute_ranking_for_reference_song(files_and_labels, model)

    # Compute aggregate metrics
    metrics = compute_mean_metrics_for_rankings(ranking, k=k)
    print(metrics)
    logging.info("Evaluation complete. Results:")
    logging.info(f"Model: {model_name}")
    logging.info(f"Mean Average Precision (mAP): {metrics['mAP']}")
    logging.info(f"Precision at {k} (mP@{k}): {metrics[f'mP@k']}")
    logging.info(f"Mean Rank of First Correct Cover (mMR1): {metrics['mMR1']}")

    return {
        "Model": model_name,
        "Mean Average Precision (mAP)": metrics["mAP"],
        f"Precision at {k} (mP@{k})": metrics[f"mP@k"],
        "Mean Rank of First Correct Cover (mMR1)": metrics["mMR1"]
    }
