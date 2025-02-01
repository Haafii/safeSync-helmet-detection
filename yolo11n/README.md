# Helmet Detection Model

## Overview

The Helmet Detection Model is a deep learning-based object detection system designed to classify images into two categories: "With Helmet" and "Without Helmet". This model utilizes the YOLOv11n architecture, known for its real-time object detection capabilities.

## Model Details

- **Functionality**: Helmet Detection
- **Classes**: Two classes - "With Helmet" and "Without Helmet"
- **Architecture**: YOLOv11n

## Dataset Description

The model is trained on a dataset comprising images labeled with the presence or absence of helmets. The dataset is balanced to ensure the model does not exhibit bias towards either class.

### 1. Dataset Composition

The dataset is divided into two main subsets: training and validation. Each subset contains images labeled according to the presence or absence of a helmet.

![Dataset Composition](/Users/hafismuhammed/Desktop/SafeSync/ML/safesync-helmet-detection/yolo11n/plots/Dataset_class_distribution.png)

- **Training Dataset**

  - **Classes**: Two classes - "With Helmet" and "Without Helmet"
  - **Distribution**: Approximately balanced, with a slight majority of images in the "With Helmet" class.
  - **Instances**: Over 35,000 instances in the "With Helmet" class and around 25,000 instances in the "Without Helmet" class.
- **Validation Dataset**

  - **Classes**: Similarly divided into "With Helmet" and "Without Helmet" classes.
  - **Distribution**: Also balanced, with a similar number of instances in each class, slightly over 4,500 for "With Helmet" and around 4,700 for "Without Helmet".

The class distribution graphs for both training and validation datasets show a balanced representation of both classes, which is crucial for training a model that does not exhibit bias towards one class over the other.

## Training Metrics

The training process involves monitoring several key metrics to ensure the model's effectiveness:

- **Precision**: Measures the accuracy of the positive predictions.
- **Recall**: Determines the model's ability to find all relevant cases within a dataset.
- **mAP50**: The mean Precision at IoU threshold of 0.5.
- **mAP50-95**: The mean Average Precision across different IoU thresholds ranging from 0.5 to 0.95.

![Model Metrics Over Time](/Users/hafismuhammed/Desktop/SafeSync/ML/safesync-helmet-detection/yolo11n/runs/detect/train/training_plots/metrics.png)

## Performance

The model achieves high precision, recall, and mAP scores, making it suitable for applications requiring accurate object detection capabilities.

![Training and Validation Losses Over Time](/Users/hafismuhammed/Desktop/SafeSync/ML/safesync-helmet-detection/yolo11n/runs/detect/train/training_plots/loss_curves.png)

## Learning Rate Schedule

![Learning Rate Schedule](/Users/hafismuhammed/Desktop/SafeSync/ML/safesync-helmet-detection/yolo11n/runs/detect/train/training_plots/learning_rates.png)

The learning rate is adjusted throughout the training epochs to optimize the convergence behavior of the model. The schedule starts with a higher learning rate that gradually decreases, facilitating a more refined model tuning towards the end of the training process.

## Loss Distributions

![Loss Distributions](/Users/hafismuhammed/Desktop/SafeSync/ML/safesync-helmet-detection/yolo11n/runs/detect/train/training_plots/loss_distributions.png)

Box loss, class loss, and DFL (Distribution Focal Loss) are crucial components of the model's training process. The distributions of these losses provide insights into the model's learning behavior and areas that might require further tuning.

## Correlation Heatmap

![Correlation Heatmap](/Users/hafismuhammed/Desktop/SafeSync/ML/safesync-helmet-detection/yolo11n/runs/detect/train/training_plots/correlation_heatmap.png)

The correlation heatmap illustrates the relationships between different training metrics. High correlations between metrics like box loss and DFL loss suggest that improvements in one area are likely to positively impact the other.

## Cumulative Training Time

![Cumulative Training Time](/Users/hafismuhammed/Desktop/SafeSync/ML/safesync-helmet-detection/yolo11n/runs/detect/train/training_plots/training_time.png)

The cumulative training time graph provides a visual representation of the time invested in training the model across epochs. This helps in understanding the resource requirements for model training.

## F1-Confidence and Precision-Confidence Curves

![F1-Confidence Curve](/Users/hafismuhammed/Desktop/SafeSync/ML/safesync-helmet-detection/yolo11n/runs/detect/train/F1_curve.png)
![Precision-Confidence Curve](/Users/hafismuhammed/Desktop/SafeSync/ML/safesync-helmet-detection/yolo11n/runs/detect/train/P_curve.png)

These curves depict the model's performance across different confidence thresholds. They are instrumental in understanding how the model's predictions vary with changes in confidence levels.

## Conclusion

The Helmet Detection Model, powered by YOLOv11n, offers a reliable solution for real-time helmet detection tasks. Its high precision, recall, and mAP scores make it suitable for applications requiring accurate object detection capabilities.
