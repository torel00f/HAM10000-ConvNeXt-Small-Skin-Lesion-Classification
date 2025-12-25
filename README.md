# HAM10000-ConvNeXt-Small-Skin-Lesion-Classification
Skin lesion classification on the HAM10000 dataset using a ConvNeXt-Small model with transfer learning. Includes data preprocessing, augmentation, class imbalance handling, and evaluation across seven diagnostic categories.
Model Description and Methodology (ConvNeXt-Small)

In this project, we implemented a skin lesion classification pipeline using the ConvNeXt-Small architecture on the HAM10000 dermatoscopic image dataset. The dataset consists of seven clinically relevant skin lesion categories: actinic keratoses and intraepithelial carcinoma (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv), and vascular lesions (vasc). In addition to images, the dataset provides metadata such as patient age, sex, lesion localization, and different types of ground truth verification.

The workflow starts with loading the metadata and mapping each image ID to its corresponding image file. All diagnostic labels are converted from text to numerical values, and the dataset is split into training (80%) and validation (20%) sets using stratified sampling to preserve class distribution. To address the strong class imbalance in the dataset, class weights are computed and applied during training.

A tf.data.Dataset pipeline is used to efficiently load, decode, resize, batch, and prefetch images. Data augmentation techniques such as random flips, rotations, zooming, brightness, and contrast adjustments are applied to improve generalization. The ConvNeXt-Small model pre-trained on ImageNet is used as the backbone, with its weights initially frozen. On top of the base model, a global average pooling layer, dropout, and a softmax classification head are added to perform seven-class classification.

The model is trained using the Adam optimizer and sparse categorical cross-entropy loss, with early stopping based on validation loss to prevent overfitting.

Results and Observations

After training, the ConvNeXt-Small model achieved a validation accuracy of approximately 55%. The weighted F1-score reached around 0.60, mainly driven by strong performance on the dominant class (melanocytic nevi, nv). This class achieved high precision and recall, indicating that the model learned its visual patterns effectively.

However, the performance on rare classes such as akiec, df, and bcc remained limited, with low recall values. Some classes, such as vascular lesions (vasc), showed very high recall but extremely low precision, suggesting over-prediction. These results highlight the impact of class imbalance and the sensitivity of high-capacity models like ConvNeXt to limited and unevenly distributed medical data.

Overall, while ConvNeXt-Small demonstrated strong representation power, its training stability and performance on minority classes were constrained by the size and imbalance of the dataset.

Comparison with DenseNet121

Compared to our earlier experiments using DenseNet121, ConvNeXt-Small showed higher capacity but lower stability. DenseNet121 achieved more balanced performance across classes and better overall reliability on this dataset, particularly for rare lesion types. In contrast, ConvNeXt-Small tended to focus heavily on the majority class, leading to reduced macro-level performance despite similar weighted metrics.

This comparison suggests that DenseNet-style architectures may be more suitable for small and imbalanced medical imaging datasets, while ConvNeXt models may require more data, stronger regularization, or more advanced fine-tuning strategies to fully realize their potential.

The implementation and results for the DenseNet121 model are provided in a separate GitHub repository for clarity and reproducibility.

Dataset

This project uses the Skin Cancer MNIST: HAM10000 dataset, available on Kaggle:
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
