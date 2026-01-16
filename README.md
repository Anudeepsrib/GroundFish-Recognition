# GroundFish-Recognition

## Cross Database and Transfer Learning Experiments with YOLOv8 Object Detection

### Overview
This repository hosts a comprehensive study on the generalization capabilities of YOLOv8 object detection models in the context of groundfish species recognition. The project utilizes machine learning and transfer learning techniques to assess model performance across disparate environments, specifically shifting between **Conveyor Belt** (controlled) and **Underwater** (wild) datasets.

The goal is to understand how well models trained in one domain adapt to another, and how transfer learning can bridge the gap.

### Key Features
*   **Cross-Domain Analysis**: Testing generalization from controlled settings to natural underwater environments.
*   **Transfer Learning**: Utilizing pre-trained weights to improve performance in target domains with limited data.
*   **YOLOv8 Implementation**: Leveraging the state-of-the-art YOLOv8 architecture for efficient and accurate detection.
*   **Roboflow Integration**: Seamless dataset management and preprocessing via Roboflow.

### Experiments Breakdown
The project consists of five core experiments, each contained in its own directory:

#### 1. [Experiment 1](./Experiment1)
*   **Generalization (Conveyor $\to$ Underwater)**
*   Train on Conveyor Belt dataset, Test on Underwater dataset.
*   *Objective*: Establish a baseline for zero-shot generalization performance.

#### 2. [Experiment 2](./Experiment2)
*   **Generalization (Underwater $\to$ Conveyor)**
*   Train on Underwater dataset, Test on Conveyor Belt dataset.
*   *Objective*: Assess if models trained in complex environments generalize better to simple ones.

#### 3. [Experiment 3](./Experiment3)
*   **Mixed Dataset Training**
*   Train and Test on a combined dataset (Conveyor + Underwater).
*   *Objective*: Evaluate if data diversity improves overall robustness.

#### 4. [Experiment 4](./Experiment4)
*   **Transfer Learning (Conveyor $\to$ Underwater)**
*   Pre-train on Conveyor Belt, Fine-tune on Underwater.
*   *Objective*: Quantify the benefits of transfer learning from a source domain.

#### 5. [Experiment 5](./Experiment5)
*   **Transfer Learning (Underwater $\to$ Conveyor)**
*   Pre-train on Underwater, Fine-tune on Conveyor Belt.
*   *Objective*: Investigate reverse transfer learning efficacy.

### Installation & Prerequisites

To replicate these experiments, you need a Python environment with GPU support (recommended).

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Anudeepsrib/GroundFish-Recognition.git
    cd GroundFish-Recognition
    ```

2.  **Install dependencies**:
    The primary dependencies are `ultralytics` (for YOLOv8) and `roboflow`.
    ```bash
    pip install ultralytics
    pip install roboflow
    ```
    *Note: Jupyter Notebook environment (like Google Colab) is recommended for running the `.ipynb` files.*

### Usage

Each experiment is self-contained in a Jupyter Notebook. To run an experiment:

1.  Navigate to the experiment folder (e.g., `Experiment1`).
2.  Open the corresponding `.ipynb` file.
3.  Ensure you have your Roboflow API key ready (if retraining or downloading datasets).
4.  Run the cells sequentially.

The notebooks include steps for:
*   Environment setup.
*   Dataset download from Roboflow.
*   Model training (or loading pre-trained weights).
*   Validation and Evaluation (Confusion Matrix, Precision-Recall curves).
*   Inference on test images.

### Results
The repository includes generated artifacts such as confusion matrices and prediction samples within the `runs/` directory of each experiment (generated during runtime). These visualizations help in analyzing:
*   **False Positives/Negatives**: Confusion between fish species or background.
*   **confidence Scores**: How confident the model is in its predictions across domains.

### Citation

If you find this project useful for your research or work, please consider citing it:

```BibTeX
@misc{Bathina-GroundFishRecognition,
  title={GroundFish Recognition: Cross Database and Transfer Learning Experiments},
  author={Bathina, Anudeepsri},
  year={2023},
  publisher={GitHub},
  journal={GitHub Repository},
  howpublished={\url{https://github.com/Anudeepsrib/GroundFish-Recognition}},
}
```
