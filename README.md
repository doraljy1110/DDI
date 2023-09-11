MultiDDI
====
Description
-----
Given the potential for harmful drug-drug interactions (DDI) in clinical settings, identifying DDIs is crucial. We have developed a deep learning-based model called MultiDDI. After being given the SMILES representations of a pair of drugs, MultiDDI outputs the specific type of DDI event by integrating the 1D, 2D, and 3D features of the drugs.

Environment
-----
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

Run MultiDDI
----
python train_ensemble.py --train-set "./Data/drugbank_train_0.csv" --batch-size 64 --epoch 20 --learn-rate 1e-4
