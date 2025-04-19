# ResNet50 Fine-Tuning

Fine-tune ResNet50 for image classification with modular PyTorch code and WandB integration.

## Structure
- `train.py`: Training script (manual & WandB sweep)
- `cnn_model.py`: ResNet50 fine-tuner class
- `datasetloader.py`: Loads dataset with transforms
- `augmentations.py`: Train/test transforms

## Dataset Format
dataset/
├── class_1/
│   └── img1.jpg
├── class_2/
│   └── img2.jpg

## Train
WandB Sweep:  
`python train.py --use_wandb ----data_dir <path to train split of data>` 

## Best Accuracy Achieved

- **Accuracy:** `87%`

## Finetuning Strategy - Only last layer of conv block and Fully connected layers are updated.
