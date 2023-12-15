# GroundFish-Recognition

Machine Learning Experiments with YOLO Object Detection
Overview
This repository contains a series of experiments conducted using the YOLO (You Only Look Once) object detection algorithm. The experiments focus on the generalization capabilities of machine learning models across various datasets, specifically in the context of groundfish species recognition.

Experiments
The experiments are structured to assess the model's performance in different environments, utilizing a combination of traditional machine learning and transfer learning techniques. Key areas of focus include cross-database generalization, model adaptability, and the efficacy of transfer learning.

Experiment 1
Objective: Evaluate the model's ability to generalize from conveyor belt to underwater environments.
Datasets: Conveyor Belt Dataset (training), Underwater Dataset (testing).
Experiment 2
Objective: Assess the model's performance when trained on underwater images and tested on a conveyor belt dataset.
Datasets: Underwater Dataset (training), Conveyor Belt Dataset (testing).
Experiment 3
Objective: Test model performance on a mixed dataset containing both underwater and conveyor belt images.
Datasets: Mixed Dataset (training and testing).
Experiment 4
Objective: Utilize transfer learning from conveyor belt to underwater dataset.
Datasets: Conveyor Belt Dataset (initial training), Underwater Dataset (fine-tuning).
Experiment 5
Objective: Apply transfer learning from underwater to conveyor belt environment.
Datasets: Underwater Dataset (initial training), Conveyor Belt Dataset (fine-tuning).
Installation
To replicate these experiments, several Python libraries are required, including ultralytics for YOLO models and roboflow for dataset management. Install these using pip:

bash
Copy code
pip install ultralytics==8.0.20
pip install roboflow
Usage
The repository includes Jupyter notebooks for each experiment with detailed code snippets for training and evaluating the models.

To run a specific experiment, navigate to the corresponding notebook and follow the instructions within.

Results
Each experiment folder contains:

Training scripts and notebooks.
Validation and testing results, including confusion matrices and prediction images.
Exported model weights and performance metrics.
