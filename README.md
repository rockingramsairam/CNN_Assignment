# Melanoma Skin Cancer Detection
 Outline a brief description of your project.
In the realm of cancer, there exist over 200 distinct forms, with melanoma standing out as the most lethal type of skin cancer among them. The diagnostic protocol for melanoma typically initiates with clinical screening, followed by dermoscopic analysis and histopathological examination. Early detection of melanoma skin cancer is pivotal, as it significantly enhances the chances of successful treatment. The initial step in diagnosing melanoma skin cancer involves visually inspecting the affected area of the skin. Dermatologists capture dermatoscopic images of the skin lesions using high-speed cameras, which yield diagnostic accuracies ranging from 65% to 80% for melanoma without supplementary technical assistance. Through further visual assessment by oncologists and dermatoscopic image analysis, the overall predictive accuracy of melanoma diagnosis can be elevated to 75% to 84%. The objective of the project is to construct an automated classification system leveraging image processing techniques to classify skin cancer based on images of skin lesions.
# Abstract
To build a CNN based model which can accurately detect melanoma. Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution which can evaluate images and alert the dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)

<!-- You can include any other section that is pertinent to your problem -->

## General Information

# Project Background:
The project aims to build a Convolutional Neural Network (CNN) based model to accurately detect melanoma, a type of skin cancer that accounts for 75% of skin cancer deaths. Early detection of melanoma can significantly reduce the manual effort needed in diagnosis and improve patient outcomes.

# Business Problem:
The primary business problem this project addresses is the need for an automated solution that can evaluate skin images and alert dermatologists about the presence of melanoma. This can help in early diagnosis, reducing the mortality rate associated with skin cancer.

# Dataset:
The dataset used in this project consists of approximately 2357 images of skin cancer types. The dataset is divided into train and test subdirectories, each containing 9 sub-directories corresponding to images of 9 different skin cancer types. The dataset is stored in Google Drive and is accessed via Google Colab for model training and evaluation.

# Conclusions

# Conclusion 1: Data Preparation and Preprocessing

The dataset was successfully loaded and preprocessed using TensorFlow and Keras libraries. Images were resized to a uniform dimension (180x180 pixels) and split into training and validation sets with an 80-20 split. This preprocessing step is crucial for ensuring that the CNN model can effectively learn from the data.

# Conclusion 2: Model Architecture

The break down of the final provided CNN architecture step by step:

Data Augmentation: The augmentation_data variable refers to the augmentation techniques applied to the training data. Data augmentation is used to artificially increase the diversity of the training dataset by applying random transformations such as rotation, scaling, and flipping to the images. This helps in improving the generalization capability of the model.

Normalization: The Rescaling(1./255) layer is added to normalize the pixel values of the input images. Normalization typically involves scaling the pixel values to a range between 0 and 1, which helps in stabilizing the training process and speeding up convergence.

Convolutional Layers: Three convolutional layers are added sequentially using the Conv2D function. Each convolutional layer is followed by a rectified linear unit (ReLU) activation function, which introduces non-linearity into the model. The padding='same' argument ensures that the spatial dimensions of the feature maps remain the same after convolution. The number within each Conv2D layer (16, 32, 64) represents the number of filters or kernels used in each layer, determining the depth of the feature maps.

Pooling Layers: After each convolutional layer, a max-pooling layer (MaxPooling2D) is added to downsample the feature maps, reducing their spatial dimensions while retaining the most important information. Max-pooling helps in reducing computational complexity and controlling overfitting.

Dropout Layer: A dropout layer (Dropout) with a dropout rate of 0.2 is added after the last max-pooling layer. Dropout is a regularization technique used to prevent overfitting by randomly dropping a fraction of the neurons during training.

Flatten Layer: The Flatten layer is added to flatten the 2D feature maps into a 1D vector, preparing the data for input into the fully connected layers.

Fully Connected Layers: Two fully connected (dense) layers (Dense) are added with ReLU activation functions. The first dense layer consists of 128 neurons, and the second dense layer outputs the final classification probabilities for each class label.

Output Layer: The number of neurons in the output layer is determined by the target_labels variable, representing the number of classes in the classification task. The output layer does not have an activation function specified, as it is followed by the loss function during training.

Model Compilation: The model is compiled using the Adam optimizer (optimizer='adam') and the Sparse Categorical Crossentropy loss function (loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)), which is suitable for multi-class classification problems. Additionally, accuracy is chosen as the evaluation metric (metrics=['accuracy']).

Training: The model is trained using the fit method with the specified number of epochs (epochs=50). The ModelCheckpoint and EarlyStopping callbacks are employed to monitor the validation accuracy during training. The ModelCheckpoint callback saves the model with the best validation accuracy, while the EarlyStopping callback stops training if the validation accuracy does not improve for a specified number of epochs (patience=5 in this case). These callbacks help prevent overfitting and ensure that the model converges to the best possible solution.

# Conclusion 4: Model Evaluation

The model's performance was evaluated on the test dataset to assess its ability to generalize to unseen data. The evaluation metrics, such as accuracy, precision, recall, and F1-score, were used to determine the model's effectiveness in detecting melanoma. The results indicated that the model achieved a high level of accuracy, making it a promising tool for assisting dermatologists in early melanoma detection.

# Conclusion 5: Future Improvements

While the model shows promising results, there is room for improvement. Future work could involve using more advanced CNN architectures, data augmentation techniques, and transfer learning to further enhance the model's performance. Additionally, expanding the dataset to include more diverse skin images could improve the model's robustness and generalization ability.
<!-- You don't have to answer all the questions - just the ones relevant to your project. -->


## Technologies Used
Python - version 3.11.11 (main, Dec  4 2024, 08:55:07) [GCC 11.4.0]
Matplotlib - version 3.10.0
Numpy - version 1.26.4
Pandas - version 2.2.2
Seaborn - version 0.13.2
Tensorflow - version 2.18.0
<!-- As the libraries versions keep on changing, it is recommended to mention the version of library used in this project -->

## Acknowledgements

UpGrad tutorials on Convolution Neural Networks (CNNs) on the learning platform
Melanoma Skin Cancer
Introduction to CNN
Image classification using CNN
Efficient way to build CNN architecture


## Contact
Created by [@rockingramsairam] - feel free to contact me!


<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

<!-- You don't have to include all sections - just the one's relevant to your project -->