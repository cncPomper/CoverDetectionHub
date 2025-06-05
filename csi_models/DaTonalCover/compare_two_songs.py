import torch
import numpy as np
import librosa
import argparse
from model import DaTonalCover

def extract_hpcp_from_mp3(mp3_path):
    y, sr = librosa.load(mp3_path, sr=44100)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma = chroma / (np.max(chroma, axis=0, keepdims=True) + 1e-6)
    return chroma.T  

def main(song1_path, song2_path, model_path="DaTonalCover.pth"):
    model = DaTonalCover()
    model.nn_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.nn_model.eval()

    hpcp1 = extract_hpcp_from_mp3(song1_path)
    hpcp2 = extract_hpcp_from_mp3(song2_path)

    sim = model.compute_tonal_similarity(hpcp1, hpcp2)
    print(f"Tonal similarity: {sim:.4f}")

    with torch.no_grad():
        pred = model.nn_model(torch.tensor([[sim]], dtype=torch.float32)).item()

    print(f"Cover prediction score: {pred:.4f}")
    if pred > 0.5:
        print("Likely a cover!")
    else:
        print("Probably not a cover.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("song1", help="Path to first song (.mp3)")
    parser.add_argument("song2", help="Path to second song (.mp3)")
    parser.add_argument("--model", default="DaTonalCover.pth", help="Path to trained model file")
    args = parser.parse_args()

    main(args.song1, args.song2, args.model)
