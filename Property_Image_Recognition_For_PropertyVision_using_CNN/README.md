**Property Image Classification with Deep Learning**
**Project Overview**
This project builds a deep learning model to classify real estate property images into categories such as backyard, bathroom, bedroom, frontyard, kitchen, and living room. The goal is to provide automated and accurate image classification for a real estate company's property listing

**Dataset**
- Labeled image dataset covering 6 property categories.
- Images resi160 160224x224 pixels.
- Data split into training and validation sets.
- Data augmentation applied: random flips, rotations, and zooms to improve generali**ion.

**Model Architecture**
- Backbone: Pre-trained ResNet50 (with imagenet weights, excluding top layers).
- Custom classification head with GlobalAveragePooling, Dense(128, ReLU), Dropout(0.4), and Dense softmax output layer.
- Initially trained with base model frozen, then fine-tuned by unfreezing last**5 layers.

**Training Details**
- Optimizer: Adam.
- Loss function: Sparse categorical cross-entropy.
- Metrics: Accuracy.
- Early stopping and learning rate reduction callbacks used.
- Initial training for 10 epochs, fine-tuning continued to 15 epochs.

**Final Performance**
- Validation Accuracy: ~74.9%
- Weighted F1-Score: ~0.74
- Good per-class precision and recall, with frontyard and **room **performing best.

**Usage**
1. Clone this repository.
2. Prepare dataset and create TensorFlow datasets (`train_ds` and `val_ds`).
3. Run the model training scripts (`train.py` or notebook).
4. Evaluate results using provided evaluation scripts for detailed classificatio**eports and learning** curves.

**Future Improvements**
- Expand training dataset and class balance.
- Try ensemble models or more complex architectures.
- Experiment with learning rate schedulers and hyperparameters.
- Furtissue or contact the project maintainer.

