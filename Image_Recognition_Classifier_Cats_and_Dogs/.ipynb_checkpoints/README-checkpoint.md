**Image Recognition Classifier**

**Project Overview**
This project implements a binary image classification model to distinguish between two classes (cats and dogs) using the TensorFlow framework with transfer learning based on the MobileNetV2 architecture. The dataset used is a filtered subset of cats and dogs images.

**Steps Performed**
**1. Import Libraries**

- Required libraries including TensorFlow, NumPy, OS, and Matplotlib were imported for deep learning, data manipulation, and visualization.

**2. Data Preprocessing**

- Downloaded and extracted the “cats and dogs” dataset from a Google storage URL.

- Defined training and validation directories.

- Loaded train and validation datasets using image_dataset_from_directory with batch size 32 and image size 160x160.

- Visualized sample images from the training set.

- Created a test set by splitting a portion (1/5) from the validation dataset.

- Configured the datasets for performance with prefetching.

**3. Data Augmentation**

- Created data augmentation layers including random horizontal flips and random rotations.

- Visualized augmented images applied to a single sample image.

**4. Model Setup with Transfer Learning**

- Applied MobileNetV2 as a pre-trained base model without the final classification layers.

- Set the input shape and used ImageNet weights.

- Froze the base model layers initially to prevent training.

- Added a global average pooling layer and dense output layer with a single neuron (binary classification).

- Created a new model combining data augmentation, preprocessing, base model, and classification head.

**5. Model Compilation and Initial Training**

- Compiled the model with Adam optimizer, binary crossentropy loss, and accuracy metrics.

- Evaluated the model on the validation dataset initially.

- Trained the model for 10 epochs on the training dataset with validation.

**6. Training Curves Visualization**

- Plotted training and validation accuracy and loss curves to evaluate performance.

**7. Fine-Tuning**

- Unfroze the top layers of the base MobileNetV2 model from layer 100 onwards.

- Recompiled the model with a lower learning rate using RMSprop optimizer.

- Continued training for 10 more epochs to fine-tune the pre-trained weights.

- Updated accuracy and loss curves were plotted to show improvements.

**8. Evaluation and Prediction**

- Evaluated the fine-tuned model on the unseen test dataset.

- Generated predictions on a batch of test images.

- Applied sigmoid to prediction logits to get probabilities and thresholded at 0.5 to obtain class labels.

- Displayed predicted labels alongside true labels.

**Summary**

This project effectively demonstrates transfer learning and fine-tuning on a small image dataset using TensorFlow’s MobileNetV2 architecture. Data augmentation enhanced generalization, and using pretrained weights accelerated convergence and improved accuracy. The model achieved good classification accuracy on test data after fine-tuning.

