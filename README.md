# Histopathology
Head and Neck SCC 



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

