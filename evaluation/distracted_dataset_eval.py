import os
import gradio as gr
import torch
import yaml
import logging
import json
from tqdm import tqdm
from csi_models.ModelBase import ModelBase
from csi_models.ByteCoverModel import ByteCoverModel
from csi_models.CoverHunterModel import CoverHunterModel
from csi_models.LyricoverModel import LyricoverModel
from csi_models.LyricoverAugmentedModel import LyricoverAugmentedModel
from csi_models.RemoveModel import RemoveModel
from feature_extraction.feature_extraction import MFCCModel, SpectralCentroidModel
from evaluation.metrics import compute_mean_metrics_for_rankings

# Load configurations
with open("configs/paths.yaml", "r") as f:
    config = yaml.safe_load(f)


DISTRACTED_DATASET_DIR = config["distracted_dataset_dir"]
DISTRACTED_DATASET_REFERENCE_DIR = config["distracted_dataset_reference_dir"]

def gather_distracted_dataset_files(dataset_path: str):
    """
    Return a list of (audio_path, label) from the distracted dataset.
    Each cover pair is given the same label.
    """
    json_path = os.path.join(dataset_path, "metadata.json")
    logging.info(f"Gathering dataset files from {json_path}")

    if not os.path.exists(json_path):
        logging.error(f"JSON file does not exist: {json_path}")
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    files_and_labels = []

    for idx, entry in enumerate(data):
        song1 = entry["song1"]
        song2 = entry["song2"]

        label = f"{song1['author']} - {song1['title']}"  # consistent label

        path1 = os.path.join(dataset_path, song1["path"])
        path2 = os.path.join(dataset_path, song2["path"])

        if not os.path.exists(path1) or not os.path.exists(path2):
            logging.warning(f"Missing files for entry {idx}: {path1}, {path2}")
            continue

        files_and_labels.extend([
            (path1, label),
            (path2, label),
        ])

    return files_and_labels


def compute_embeddings(files_and_labels, model: ModelBase, progress=gr.Progress()):
    """Compute and cache embeddings for all audio files using the given model."""
    progress(0, desc="Calculating embeddings")
    embeddings = {}
    total_files = len(files_and_labels)

    for i, (audio_path, _) in enumerate(tqdm(files_and_labels, desc="Computing embeddings")):
        if audio_path not in embeddings:
            embeddings[audio_path] = model.compute_embedding(audio_path)
        progress((i + 1) / total_files, desc="Calculating embeddings")

    return embeddings


def compute_rankings_per_song(files_and_labels, model: ModelBase, progress=gr.Progress()):
    """Compute rankings for each song in the dataset."""
    logging.info("Starting computation of rankings per song.")
    embeddings = compute_embeddings(files_and_labels, model)
    rankings_per_query = []
    total_files = len(files_and_labels)
    progress(0, desc="Comparing embeddings")

    for i, (query_path, query_label) in enumerate(tqdm(files_and_labels, desc="Processing queries")):
        query_embedding = embeddings[query_path]
        if query_embedding is None:
            logging.warning(f"No embedding found for query file {query_path}. Skipping.")
            continue
        comparisons = []
        for j, (cand_path, cand_label) in enumerate(files_and_labels):
            if i == j:
                continue

            cand_embedding = embeddings[cand_path]
            sim = model.compute_similarity(query_embedding, cand_embedding)
            gt = (query_label == cand_label)

            comparisons.append({"candidate_path": cand_path, "similarity": sim, "ground_truth": gt})

        comparisons.sort(key=lambda x: x["similarity"], reverse=True)

        rankings_per_query.append({
            "query_path": query_path,
            "query_label": query_label,
            "ranking": comparisons
        })
        progress((i + 1) / total_files, desc="Comparing embedding pairs")

    return rankings_per_query


def evaluate_on_distracted_dataset(model_name: str, reference=False, k=10):
    """Evaluate a model on the distracted dataset."""
    if reference:
        dataset_path = DISTRACTED_DATASET_REFERENCE_DIR
    else:
        dataset_path = DISTRACTED_DATASET_DIR
    logging.info(f"Evaluating model '{model_name}' on dataset '{dataset_path}' with k={k}.")
    files_and_labels = gather_distracted_dataset_files(dataset_path)

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

    rankings_per_query = compute_rankings_per_song(files_and_labels, model)
    metrics = compute_mean_metrics_for_rankings(rankings_per_query, k=k)

    if reference:
        return {
            "Model": model_name,
            "Dataset": "distracted_dataset reference",
            "Mean Average Precision (mAP)": metrics["mAP"],
            f"Precision at {k} (mP@{k})": metrics["mP@k"],
            "Mean Rank of First Correct Cover (mMR1)": metrics["mMR1"]
        }
    else:
        return {
            "Model": model_name,
            "Dataset": "distracted_dataset",
            "Mean Average Precision (mAP)": metrics["mAP"],
            f"Precision at {k} (mP@{k})": metrics["mP@k"],
            "Mean Rank of First Correct Cover (mMR1)": metrics["mMR1"]
        }
