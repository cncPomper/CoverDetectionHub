import logging
import model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import numpy as np

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    csv_path = "datasets\datacos\da_tacos_listed.csv"        
    hpcp_base = r"D:\WimuProj\da-tacos_benchmark_subset_hpcp" # to ma 10 gb ten folder temu to jest sciezka do dysku a nie projektowa

    daTonal = model.DaTonalCover()

    pairs = daTonal.generate_pairs_from_csv(csv_path,0) # 0 for no limits
    #print(pairs)
    logging.info(f"Generated {len(pairs)} pairs.")
  
    X, y = daTonal.extract_similarity_features(pairs, hpcp_base)
    logging.info(f"Extracted features for {len(X)} pairs.")

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model_path = "DaTonalCover.pth"
    trained_model = daTonal.train_tonal_model(X_train, y_train, epoch_number=5, model_path=model_path)
    logging.info("Model training complete and saved.")

    # Evaluate on test set
    with torch.no_grad():
        daTonal.nn_model.eval()
        preds = daTonal.nn_model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
        preds_bin = (preds > 0.5).astype(int)
        acc = accuracy_score(y_test, preds_bin)
        print(f"Test Accuracy: {acc:.4f}")
