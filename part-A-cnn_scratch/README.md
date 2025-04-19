# CNN Classifier on iNaturalist Dataset

## Dataset

The project uses the **iNaturalist** dataset, a large-scale image classification dataset focused on species of animals, plants, and fungi. Each image is labeled with its corresponding category, and the dataset is typically used for fine-grained classification tasks. In this implementation, the images are organized in a directory structure compatible with `torchvision.datasets.ImageFolder`, making it easy to load and preprocess.

## Code Structure

- **`cnn_model.py`** – Defines a customizable CNN architecture built from scratch using PyTorch, including convolutional layers, batch normalization, activations, and fully connected layers.
- **`train.py`** – Manages training, validation, argument parsing, and integration with Weights & Biases (W&B) for logging and sweep support.
- **`datasetloader.py`** – Loads the dataset, applies transforms, and handles stratified splitting for training and validation sets.

All hyperparameters like number of layers, kernel size, activation functions, and dense layer sizes are configurable, and W&B sweeps are used for systematic experimentation.

## Best Accuracy Achieved

- **Accuracy:** `39.25%`
- **Augmentation:** Not applied yet
- **Sweep:** Conducted with varying configurations to reach this baseline

