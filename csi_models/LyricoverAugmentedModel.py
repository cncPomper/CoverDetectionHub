import os
import sys
import logging
import torch
import yaml

lyricover_path = os.path.abspath("./csi_models/lyricover")
if lyricover_path not in sys.path:
    sys.path.insert(0, lyricover_path)

from csi_models.ModelBase import ModelBase
from csi_models.lyricover.utils import load_whisper_model
from csi_models.lyricover.model import CoverClassifier

class LyricoverAugmentedModel(ModelBase):
    def __init__(self, config_path="configs/paths.yaml", device=None):
        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.checkpoint_path = config["lyricover_augmentation_checkpoint_path"]
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Add Lyricover directory to sys.path
        lyricover_path = os.path.abspath("./csi_models/lyricover")
        if lyricover_path not in sys.path:
            sys.path.insert(0, lyricover_path)

        # Load the model
        self.model = self._load_model()

    def _load_model(self, instrumental_threshold=8, lyrics_model=None):
        logging.info("Loading Lyricover model...")

        if lyrics_model is None:
            logging.info("Loading default Whisper model...")
            lyrics_model = load_whisper_model()

        classifier = CoverClassifier(
            instrumental_threshold=instrumental_threshold,
            lyrics_model=lyrics_model
        )

        classifier.load_model(self.checkpoint_path)
        logging.info("Lyricover model loaded and ready for inference.")

        return classifier

    def compute_similarity_between_files(self, audio1_path, audio2_path):
        logging.info(f"Computing Lyricover similarity for:\n  {audio1_path}\n  {audio2_path}")

        lyrics_1, is_instrumental_1, tonal_features_1 = self.model.calculate_song_features(audio1_path)
        lyrics_2, is_instrumental_2, tonal_features_2 = self.model.calculate_song_features(audio2_path)

        prediction = self.model.compute_similarity_and_predict(
            tonal_features_1, tonal_features_2, lyrics_1, lyrics_2, is_instrumental_1, is_instrumental_2
        )

        logging.info(f"Prediction score (cover likelihood): {prediction:.4f}")
        return prediction

    def compute_embedding(self, audio_path):
        logging.info(f"Computing Lyricover embedding for:\n  {audio_path}")

        lyrics, is_instrumental, tonal_features = self.model.calculate_song_features(audio_path)

        logging.info("Lyricover embedding computed.")
        return (lyrics, is_instrumental, tonal_features)

    def compute_similarity(self, embedding1, embedding2):
        logging.info("Computing Lyricover similarity...")

        lyrics_1, is_instrumental_1, tonal_features_1 = embedding1
        lyrics_2, is_instrumental_2, tonal_features_2 = embedding2

        prediction = self.model.compute_similarity_and_predict(
            tonal_features_1, tonal_features_2, lyrics_1, lyrics_2, is_instrumental_1, is_instrumental_2
        )

        logging.info(f"Prediction score (cover likelihood): {prediction:.4f}")
        return prediction

# Example usage:
# lyricover = LyricoverModel()
# similarity = lyricover.compute_similarity_between_files("audio1.wav", "audio2.wav")
# embedding = lyricover.compute_embedding("audio1.wav")
