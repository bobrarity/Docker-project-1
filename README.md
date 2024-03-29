# Iris Classification Project

## Features
- Data processing script for the Iris dataset.
- Dockerized environment for training and inference.
- PyTorch model for classification.

## Prerequisites
- Docker
- Python
- Python libraries

## Project Structure
- `training/`: Training scripts and Dockerfile.
- `inference/`: Inference scripts and Dockerfile.
- `shared_folder/`: Contains PyTorch model and model predictions.
- `.gitignore`: Specifies untracked files.
- `README.md`: Project description and instructions.

## Setup and Running
- Download the ZIP file of repo
- Create a folder named shared_folder in main folder
### Training
First, navigate to the 'training' directory.
Build the Docker image for training: 
- docker build -t iris_training_image .
    
Run the training container:
- docker run -v /path/to/shared_folder:/app/shared_folder iris_training_image

### Inference
First, navigate to the 'inference' directory.
Build the Docker image for inference:
- docker build -t iris_inference_image .

Run the inference container:
- docker run -v /path/to/shared_folder:/app/shared_folder iris_inference_image
