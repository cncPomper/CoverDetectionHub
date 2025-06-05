### Dataset: Covers80 

| Dataset / Model          | Mean Average Precision (mAP) | Precision at 10 (P@10) / mP@10 | Mean Rank of First Correct Cover (MR1 / mMR1) |
|--------------------------|------------------------------|---------------------------------|-----------------------------------------------|
| **Model: Lyricover**     | 0.83425                     | 0.09939                         | 7.41463                                       |
| **Model: Lyricover on bigger dataset (8k)**  | 0.77214                    | 0.07215                        | 5.53153                                     |
| **Model: Lyricover on bigger dataset (4k)**  | 0.79251                    | 0.07984                    | 6.21521
| **Model: Lyricover on bigger dataset (2k)**  | 0.81652                  | 0.08812                        | 6.95124    

![alt text](image.png)

## Why Larger Dataset Didn’t Improve Lyricover

- **Data Quality vs. Quantity**  
  Additional examples likely introduced noise (mismatched genres, poor recordings) that diluted the model’s signal.

- **Limited Feature Space**  
  Using only tonal cosine similarity and basic lyric cosine similarity cannot capture nuanced cover relationships as data diversity grows.

- **Model Capacity Mismatch**  
  A small 3-layer NN may underfit on more varied data unless re-architected or re-tuned (learning rate, regularization).

- **Imbalanced Sampling**  
  Adding many negatives without balancing positives can push down precision and mAP, even if overall accuracy remains stable.

---

## Concise Future Directions

1. **Richer Features**  
   - Use multi-scale tonal embeddings (e.g., short-term and long-term chroma).  
   - Replace raw lyric cosine with contextual text embeddings (e.g., Sentence-BERT).

2. **Stronger Models & Losses**  
   - Adopt a dual-branch network: one branch for audio embeddings (e.g., small CNN on mel spectrogram), another for lyric embeddings; fuse before classification.  
   - Switch to contrastive or triplet loss so covers cluster in embedding space, boosting mAP and P@10 directly.

3. **Data Curation & Sampling**  
   - Filter out noisy/mislabeled pairs automatically (e.g., check vocal separation confidence, validate lyric alignment).  
   - Ensure balanced positive vs. hard-negative sampling (hard negatives are non-covers with similar chords or themes).

4. **Transfer Learning & Pretrained Embeddings**  
   - Leverage pretrained audio models (e.g., OpenL3, VGGish) for richer audio features.  
   - Pretrain a small multimodal network (audio + lyrics) on a larger external covers dataset, then fine-tune on Covers80.

5. **Rank-Aware Objectives**  
   - Optimize a differentiable ranking loss (e.g., soft mAP) to directly improve P@10.  
   - Use margin ranking loss to minimize the mean rank of the first correct cover (MR1).
