## Sports Image Classification

### Overview

This project aims to classify sports images into various categories using advanced deep learning models. We leverage Convolutional Neural Networks (CNNs) and transfer learning techniques to build robust image classification models. The dataset used is the Sports Classification dataset from Kaggle, which contains images of different sports categories split into training and test sets.

### Dataset

The dataset comprises images organized into directories by class. The training and test sets are used to train the models and evaluate their performance. The dataset is structured as follows:

- `train/`: Directory containing training images organized by class.
- `test/`: Directory containing test images organized by class.

### Data Preprocessing

Data preprocessing is a crucial step to prepare the images for training the models. The steps include:

1. **Image Augmentation**: Techniques such as rotation, shifting, shearing, zooming, and horizontal flipping are applied to increase the diversity of the training data and improve the model's ability to generalize.
2. **Normalization**: Image pixel values are normalized to the range [0, 1] to make the training process more stable and efficient.
3. **Data Generators**: `ImageDataGenerator` is used to create batches of augmented and normalized images for training and validation.

### Models Used

We used three advanced models for image classification: ResNet50, InceptionV3, and EfficientNetB0. These models leverage transfer learning from pre-trained networks on the ImageNet dataset, which allows us to benefit from the features learned on a large and diverse dataset.

#### Model 1: ResNet50

**ResNet50** is a deep residual network with 50 layers. Residual networks introduce shortcut connections that bypass one or more layers, solving the problem of vanishing gradients in deep networks and allowing for very deep architectures.

**Architecture Highlights**:
- **Residual Blocks**: Allow gradients to flow directly through the network, making it easier to train very deep networks.
- **Pre-trained on ImageNet**: The base model is pre-trained on the ImageNet dataset, providing a solid foundation for our classification task.

**Performance**:
ResNet50 demonstrated strong performance, leveraging its depth to capture complex features in the images. The pre-trained weights allowed for efficient transfer learning, leading to high accuracy on the sports classification task.

#### Model 2: InceptionV3

**InceptionV3** is an advanced CNN architecture that employs inception modules to capture multi-scale features efficiently. Inception modules use multiple convolutional filters of different sizes in parallel, allowing the network to learn features at various scales.

**Architecture Highlights**:
- **Inception Modules**: Enable the network to capture diverse features by combining filters of different sizes.
- **Factorized Convolutions**: Reduce computational cost by factorizing larger convolutions into smaller ones.
- **Pre-trained on ImageNet**: Utilizes the extensive feature set learned from the ImageNet dataset.

**Performance**:
InceptionV3 performed well on the sports classification task, with its inception modules effectively capturing the variations in the sports images. The network's ability to process images at multiple scales contributed to its robust performance.

#### Model 3: EfficientNetB0

**EfficientNetB0** is a state-of-the-art model that balances network depth, width, and resolution to achieve high performance with fewer parameters. EfficientNet uses a compound scaling method to uniformly scale all dimensions of the network.

**Architecture Highlights**:
- **Compound Scaling**: Efficiently balances depth, width, and resolution for optimal performance.
- **Lightweight and Efficient**: Achieves high accuracy with fewer parameters compared to other models.
- **Pre-trained on ImageNet**: Benefits from the rich features learned from the ImageNet dataset.

**Performance**:
EfficientNetB0 achieved excellent results on the sports classification task, with its efficient architecture providing a good balance between accuracy and computational cost. The model's design allowed it to perform well even with fewer parameters.

### Training and Evaluation

Each model was trained using the training data generator and validated using the test data generator. The training process involved:

- **Training for 10 epochs**: Each model was trained for 10 epochs, which was sufficient to observe convergence in accuracy and loss metrics.
- **Monitoring Accuracy and Loss**: Training and validation accuracy and loss were monitored to evaluate model performance and detect overfitting.

### Results and Conclusion

All three models performed well on the sports classification task, with EfficientNetB0 providing the best balance between accuracy and efficiency. The use of transfer learning significantly boosted the performance of all models, leveraging the rich feature sets learned from the ImageNet dataset.

**Key Observations**:
- **ResNet50**: High accuracy due to deep architecture and residual connections.
- **InceptionV3**: Strong performance with efficient multi-scale feature extraction.
- **EfficientNetB0**: Excellent accuracy with fewer parameters, making it the most efficient model.

### Conclusion

The project successfully demonstrated the effectiveness of using advanced deep learning models and transfer learning for sports image classification. EfficientNetB0 emerged as the best model, providing high accuracy with optimal efficiency. The use of image augmentation and normalization further enhanced the model's ability to generalize to new data.

This project showcases the potential of modern deep learning techniques in solving complex image classification tasks and highlights the benefits of leveraging pre-trained models for efficient and accurate performance.

---


## Dataset

The dataset used for this project is the Sports Classification dataset available on Kaggle. The dataset contains images of different sports categories, which are split into training and test sets.

## Project Structure

- `train/`: Directory containing training images organized by class.
- `test/`: Directory containing test images organized by class.
- `notebook.ipynb`: Jupyter Notebook with code for data preprocessing, model training, and evaluation.
- `README.md`: Documentation of the project.

## Steps Performed

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/sports-image-classification.git
    cd sports-image-classification
    ```

2. **Download the dataset**:
    - Go to [Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification) and download the dataset.
    - Unzip the dataset and place the `train` and `test` directories in the root of the repository.

3. **Run the Jupyter Notebook**:
    Open `notebook.ipynb` and run all cells to preprocess the data, train the models, and evaluate the results.

## Dependencies

- TensorFlow
- Keras
- Matplotlib
- Pandas
- Numpy

Install the dependencies using:
```bash
pip install tensorflow keras matplotlib pandas numpy
```

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
