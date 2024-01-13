MDFF:A deep learning model for drug-drug interaction prediction
====
# Breif introduction
Concurrent use of multiple drugs can lead to unexpected drug-drug interactions (DDIs), potentially resulting in adverse clinical outcomes. MDFF, a multi-dimensional feature fusion model was desigined for DDIs prediction.By incorporating drug information across three different dimensions(1D sequence, 2D topological, and 3D geometry), MDFF is capable of capturing the complete physicochemical characteristics of drugs, providing the most accurate and holistic view of drug interactions.
# Environment
python=3.9.0
pandas=2.0.3
numpy=1.24.1
matplotlib=3.7.2
scikit-learn=1.3.0
rdkit=2023.3.3
torch=2.0.1+cu118
torch_geometric=2.3.1
torch_scatter=2.1.1+pt20cu118
torchmetrics=0.10.2
# How to run MDFF
python train_ensemble.py --train-set "./Data/drugbank_train_0.csv" --batch-size 64 --epoch 20 --learn-rate 1e-4
