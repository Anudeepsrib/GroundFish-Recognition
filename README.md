# GroundFish-Recognition

## Cross Database and Transfer Learning Experiments with YOLOv8 Object Detection

### Overview
This repository contains a series of experiments conducted using the YOLO (You Only Look Once) object detection algorithm. The experiments focus on the generalization capabilities of machine learning models across various datasets, specifically in the context of groundfish species recognition.

### Experiments
The experiments are structured to assess the model's performance in different environments, utilizing a combination of traditional machine learning and transfer learning techniques. Key areas of focus include cross-database generalization, model adaptability, and the efficacy of transfer learning.

![image](https://github.com/Anudeepsrib/GroundFish-Recognition/assets/36981925/97aa6677-4f65-44ba-a586-79cb576955e1)


#### Experiment 1
- **Objective**: Evaluate the model's ability to generalize from conveyor belts to underwater environments.
- **Datasets**: Conveyor Belt Dataset (training), Underwater Dataset (testing).

#### Experiment 2
- **Objective**: Assess the model's performance when trained on underwater images and tested on a conveyor belt dataset.
- **Datasets**: Underwater Dataset (training), Conveyor Belt Dataset (testing).

#### Experiment 3
- **Objective**: Test model performance on a mixed dataset containing underwater and conveyor belt images.
- **Datasets**: Mixed Dataset (training and testing).

#### Experiment 4
- **Objective**: Utilize transfer learning from conveyor belt to underwater dataset.
- **Datasets**: Conveyor Belt Dataset (initial training), Underwater Dataset (fine-tuning).

#### Experiment 5
- **Objective**: Apply transfer learning from underwater to a conveyor belt environment.
- **Datasets**: Underwater Dataset (initial training), Conveyor Belt Dataset (fine-tuning).

### Installation
Several Python libraries are required to replicate these experiments, including Ultralytics for YOLO models and Roboflow for dataset management. Install these using pip:

```bash
pip install ultralytics==8.0.20
pip install roboflow
