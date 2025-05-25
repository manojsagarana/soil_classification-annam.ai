Soil-Classification-Challenge
Soil Image Classification Challenge (IIT Ropar, Annam.ai)
This repository contains our complete solutions to two soil image classification challenges hosted by Annam.ai at IIT Ropar. The primary goal was to develop high-performing deep learning models to accurately classify soil images, maximizing F1-score for balanced predictions.

Project Structure
Soil Classification Part 1/
Multiclass classification of soil images into four types: Alluvial, Black, Clay, and Red.

Soil Classification Part 2/
Binary classification: Soil vs. Non-Soil images, with special focus on handling class imbalance.

Each folder contains:

All code, Jupyter notebooks, and README.md for that specific challenge.
Example requirements and setup instructions.
Datasets and submission templates (see inside respective folders).

Team Members
Team Name - Team JM
Manoj Sagaran. A, VIT Chennai, manojsagaran.a2022@vitstudent.ac.in
Joshika. B R, VIT Chennai, joshika.br2022@vitstudent.ac.in

Highlights
Challenge 1: Achieved a perfect F1-score of 1.0 across all soil classes using a fine-tuned ResNet18 model.
Challenge 2: Addressed class imbalance by manually augmenting the dataset with non-soil images, significantly improving generalization and reducing false positives.

Directory Structure
Soil-Classification/
│
├── Soil Classification Part 1/
│   ├── submission.csv
│   └── soil-classification-ch1.ipynb
│
├── Soil Classification Part 2/
│   ├── submission.csv
│   └── soil-classification-ch2.ipynb
│
└── README.md

Project Logic & Approach

Challenge 1: Four-Class Soil Classification
Preprocessing:
All images resized to 128x128.
Applied augmentations: flips, rotation, normalization.
Dataset:
Custom PyTorch SoilDataset class, labels from CSV.
Model:
Pretrained ResNet18, last layer adapted for 4 classes.
Training:
CrossEntropyLoss, Adam optimizer, F1-score monitored per class.
Best model checkpointed based on validation F1.
Prediction:
Test set inference, submission CSV generated.

Challenge 2: Soil vs. Non-Soil (Binary Classification)
Problem:
Binary task (1: Soil, 0: Not Soil).
Training set had only soil images; non-soil images added manually for generalization.
Model:
Pretrained ResNet18, last FC layer changed to binary output.
BCEWithLogitsLoss, Adam optimizer.
Augmentation:
Flips, color jitter, rotation.
Evaluation:
F1-score used for model selection and validation.
Submission:
submission.csv with image_id and predicted label.
