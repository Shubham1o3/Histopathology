# Histopathology
Head and Neck SCC
<p align="center">
<img width="1002" height="402" alt="image" src="https://github.com/user-attachments/assets/a842e973-5e34-46d9-a30d-3ca1974f3c12" />
</p>


## Model 1: ABMIL-Vanilla model 
Attention based Multi Instance Learning(ABMIL) is a weakly supervised learning model for classification and segmentation.
It is used for gigapixel images classification like whole slide images.

Files and their Purpose
1. main.py : main script for training model (Modify hyperparameters learning rate, No. of epochs)
2. Inference.py = script to find classification(prediction) of wsi by giving slide name.
3. Heatmap.py = scrpit to visualize heatmap and predction of wsi.

 Folder :
1. Models : store MIL model (ABMIL, can add other models)
2. Utils : contains scripts for dataloader, train, inference and heatmap

Current Result :
1. N0 dataset : FN - 1 , Fp - 0 (Test loss: 0.1045      Test acc: 0.9844    Test auc: 0.9918)
2. CamelYon17 : model overfits for ABMIL-plain model

## Model 2: ABMIL Pseudo Augmentation model (ABMIL-Pse)
- ABMIL-Pse is a data augmentation technique designed to create pseudo bags by sampling a subset of instances from whole slide images (WSIs). 
- The process begins by extracting features from each instance of a WSI and grouping them into K phenotype clusters using cosine similarity, ensuring that similar features are clustered together. 
- From these clusters, N pseudo bags are constructed, where each bag contains a few instances from each cluster. Additionally, instances from multiple WSIs can be mixed into a single pseudo bag to increase diversity. 
This strategy helps enrich the training data and reduces the risk of overfitting in weakly supervised learning models.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f39f625b-04d5-4439-8772-9633bb7a23ac" alt="abmil-pse" width="500" height="500" />
  <br>
  <em>Figure : Pipelne for ABMIL-Pse</em>
</p>


<p align="center">
  <img src="https://github.com/user-attachments/assets/0918f4d1-0a98-4378-b2c7-d4ba182b060f" alt="pse" width="600" height="616" />
  <br>
  <em>Figure : Pseudo bag mixup method </em>
</p>


Files and their Purpose
1. main.py : main script for training model (Modify hyperparameters learning rate, No. of epochs)
2. Inference.py = script to find classification(prediction) of wsi by giving slide name.
3. Heatmap.py = scrpit to visualize heatmap and predction of wsi.

 Folder :
1. Models : store MIL model (ABMIL, can add other models)
2. Utils : contains code for dataloader, train, inference and heatmap
3. pse_utils : contains pseudo augmentation code

Current Result :
1. N0 dataset : FN - 3 , FP - 3 
2. CamelYon17 : FN - 2, FP - 0

Classification and segmentation:
1. N0 dataset:
   
<p align="center">
<img width="700" height="350" alt="image" src="https://github.com/user-attachments/assets/17a158a3-66e4-4cff-b75c-6f7dbb67eaba" />
</p>

2. Camelyon 17:
   ABMIL-Pse
<p align="center">
  <img height="400" alt="patient_017_node_4" src="https://github.com/user-attachments/assets/12eb5bc1-4d7f-4e38-bf31-55a96f67c341" />
  <img src="https://github.com/user-attachments/assets/87be550e-a1af-457c-954f-fbc907148f16" alt="patient_015_node_2" height="400" />
</p>

- Ground truth identification on Camelyon17 (Green:ground truth, Red:prediction)
<p align="center">
  <img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/c744df44-2bef-4f27-8e74-af42e163e166" />
</p>
