from feature_extraction.audio_preprocessing import validate_audio, InvalidMediaFileError
from evaluation.covers80_eval import evaluate_on_covers80
from evaluation.abracadabra_eval import evaluate_on_injected_abracadabra
from evaluation.distracted_dataset_eval import evaluate_on_distracted_dataset
from feature_extraction.feature_extraction import MFCCModel, SpectralCentroidModel
from csi_models.ByteCoverModel import ByteCoverModel
from csi_models.CoverHunterModel import CoverHunterModel
from csi_models.LyricoverModel import LyricoverModel
from csi_models.LyricoverAugmentedModel import LyricoverAugmentedModel
from csi_models.RemoveModel import RemoveModel
from generator.generate_cover import generate_cover
from generator.generate_cover_with_lyrics import generate_cover_with_lyrics


def gradio_cover_interface(audio1, audio2, model_name, threshold):
    """
    Gradio interface function for checking if two audio files are covers of each other.
    """
    try:
        # Validate and preprocess audio files
        audio1, error1 = validate_audio(audio1)
        if error1:
            return f"Error with Query Song: {error1}", ""

        audio2, error2 = validate_audio(audio2)
        if error2:
            return f"Error with Potential Cover Song: {error2}", ""
    except InvalidMediaFileError as e:
        return str(e), ""

    # Load the appropriate model
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
        return "Invalid model selected.", ""

    model = model_mapping[model_name]()

    # Compute similarity using the selected model
    embedding1 = model.compute_embedding(audio1)
    embedding2 = model.compute_embedding(audio2)
    similarity = model.compute_similarity(embedding1, embedding2)

    # Determine cover or not based on similarity & threshold
    result = "Cover" if similarity >= threshold else "Not a Cover"
    return result, f"Similarity Score: {similarity}"

def gradio_test_interface(model_name, dataset):
    """
    Gradio interface function for testing a model on a given dataset (Covers80, etc.).
    """
    if dataset == "Covers80":
        results = evaluate_on_covers80(model_name)
    elif dataset == "Covers80but10":
        results = evaluate_on_covers80(model_name, covers80but10=True)
    elif dataset == "Injected Abracadabra":
        results = evaluate_on_injected_abracadabra(model_name)
    elif dataset == "distracted_dataset":
        results = evaluate_on_distracted_dataset(model_name)
    elif dataset == "distracted_dataset_reference":
        results = evaluate_on_distracted_dataset(model_name, reference=True)
    else:
        return "Invalid dataset selected."

    summary_table = (
        f"Mean Average Precision (mAP): {results.get('Mean Average Precision (mAP)', 'N/A')}\n"
        f"Precision at 10 (P@10): {results.get('Precision at 10 (mP@10)', 'N/A')}\n"
        f"Mean Rank of First Correct Cover (MR1): {results.get('Mean Rank of First Correct Cover (mMR1)', 'N/A')}\n"
    )
    return summary_table

def gradio_generate_cover_interface(audio_filepath: str, duration: int, cover_type: str, alpha: float ) -> str:
    if cover_type == "Instrumental":
        return generate_cover(audio_filepath, duration)
    else:
        return generate_cover_with_lyrics(
            audio_path=audio_filepath,
            duration=duration,
            alpha=alpha
        )

