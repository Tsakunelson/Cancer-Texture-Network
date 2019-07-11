# Cancer-Texture-Network
Texture-Based Deep Learning for Histopathology Cancer Whole Slide Image (WSI) Classification

1. Installation
   - OpenSlide  
   - Pytorch
   - Torchvision
   - Skorch
   - Sklearn
   - Matplotlib
   - Keract
   - Python 3.6
   
   System Settings
   - Operating System: Linux Server (Ubuntu 18.0)
   - CPU: Intel(R) Xeon(R) CPU E5-2670 v3 @ 2.30GHz
   - GPU: NVIDIA-SMI, Driver Version: 410.48 
   - RAM: 503G
   - Storage:1T
   

2. Project Motivation

Automatic histopathological Whole Slide Image (WSI) analysis for cancer classification has been highlighted along with the advancements in microscopic imaging techniques. Manual examination and diagnosis with WSIs is time-consuming and tiresome. Recently, deep convolutional neural networks have succeeded in histopathological image analysis. In this project, we propose a novel cancer texture-based deep neural network (CAT-Net) that learns scalable texture features from histopathological WSIs. The innovation of CAT-Net is twofold: (1) capturing invariant spatial patterns by dilated convolutional layers and (2) Reducing model complexity while improving performance. Moreover, CAT-Net can provide distinctive texture patterns formed on cancerous regions of histopathological images compared to normal regions.


3. Dataset Description
![Data Samples](https://github.com/Tsakunelson/Cancer-Texture-Network/blob/master/Slide1.PNG)
We obtained gastroscopic biopsy specimen of 94 cases at the Gyeongsang National University Changwon Hospital (Changwon, Korea) between February 2016 and July 2017, and the tissue specimens were stained with hematoxylin and eosin (H&E) using standard protocols in routine clinical care. This study included 188 whole slide images (WSIs) with 26, 22, 40, 40 and 60 WSIs for well, moderately differentiated adenocarcinoma, poorly differentiated adenocarcinoma, poorly cohesive carcinoma including signet-ring cell features, and normal gastric mucosa, respectively. The histologic type and differentiation grade of the carcinoma was determined according to the classification system of the World Health Organization, fourth edition. This study was approved by the Institutional Review Board of Gyeongsang National University Hospital with a waiver for informed consent (2018-08-005-001). Out of the 188 WSIs, half (144 slides) are made of H&E stains, and the other half consist of identical slides, but made of CK stains as shown in Figure 19 obtained with Aperio image scope visualizer. For H&E slides, ~249k patches are extracted, with ~162k normal patches and ~87k tumor patches as explained in section 3.3. To promote a fair environment between both classes, we select all the ~87k tumor patches, and perform a random selection without replacement of ~87k out of ~162k equivalent normal patches. We split the resulting data into a train, validation and test ratio of 2:1:1.


4.File Descriptions

With respect to the framework, This repository is partitioned into three main sections:
1. Data Preprocessing
   - load_svs.py
   - load_good_svs.py
2. Training
   - load_data.py
   - gridSearchSkorch.py
   - cat_net_model.py
   - cat_net_model_spatial_pyramid_pooling.py
3. Inferencing
   - Predict.py
   - Heat_map.py

5. How To Interact With Your Project 

This project follows the Cross Industry Standard Process for Data Mining (CRISP-DM) by asking and answering analystical questions from Digital Pathologist. It adopts the Extract Trasform Load strategy to build pipelines ready made for deployment. A Medium Blog post on the following link describes the data loading process with pythorch for our model: [Link](https://medium.com/@tsakunelsonz/loading-and-training-a-neural-network-with-custom-dataset-via-transfer-learning-in-pytorch-8e672933469?source=friends_link&sk=587f18bded4163d4458939fd97563b96)

This is A Novel Deep learning Cancer Texture Network (CAT-Net) for Cancer detection. I this project, Digital Pathology analysis using Deep Learning models is leveraged on Whole Slide Image (WSI) Biopsy with Pytorch (and Skorch) as framework. Detection and classification of cancerous cells with a heat map visual is adopted, to enhance automatic and accurate decision making by pathologist. Through the CRISP-DM model and ETL pipelines, an end-to-end framework is created, ready made for production. 


6. Authors, Acknowledgements

Author

Nelson Zange TSAKU 

Acknowledgements 

  Much appreciations to Data X Lab for hands of advise and Orientation
  
  Much appreciations to the Gyeongsang National University Changwon Hospital of Korea for providing relevant private datasets (Informed consent 2018-08-005-001)
