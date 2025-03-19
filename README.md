# CAPTCHA Solver

## Description
CAPTCHA Solver is a tool for automatically recognizing alphanumeric CAPTCHAs using a YOLOv5-based neural network model.

## Features
- Solves alphanumeric CAPTCHAs
- Uses a trained model on a custom dataset
- Built with PyTorch and YOLOv5
- Supports batch image processing

## Requirements
- Python 3.9+
- Torch 2.4.1+
- OpenCV
- Ultralytics YOLOv5
- Pytorch

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/kirirut/Captcha-solver.git
   cd Captcha-solver
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Place CAPTCHA images in the `test` folder
2. Run the script:
   ```sh
   python captchareg.py
   ```
3. Recognized results will be saved in the `output` folder

## Project Structure
```
|-- Captcha-solver/
    |-- model/
        |-- best.pt  # Trained model
    |-- test/        # Folder with test images
    |-- output/      # Folder for results
    |-- captchareg.py  # Main script
    |-- requirements.txt  # Dependencies
    |-- README.txt  # Project description
```


