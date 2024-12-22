# Enhancing Smartphone-Based Pedestrian Dead Reckoning Using AI in Various Motions

This project aims to improve the accuracy of smartphone-based Pedestrian Dead Reckoning (PDR) systems by utilizing various smartphone sensors and LSTM networks. <br>
Our research focuses on maintaining consistent accuracy across different smartphone usage motions.

## 📑 Paper Submission
This research has been submitted to the 2024 IPNT Conference 

## 📋 Project Overview
Pedestrian Dead Reckoning (PDR) systems often suffer from accuracy issues due to varying smartphone usage motions. 
This project proposes an LSTM-based approach to enhance PDR performance in three distinct motions:

Holding the phone in front (보고 걷기) <br>
Swinging the phone while walking (흔들며 걷기) <br>
Walking with the phone in the pocket (주머니에 넣고 걷기) <br>

Key Features
Collected 4,000 steps of data for each motion at a 50Hz sampling rate. <br>
Trained an LSTM model with different time steps, achieving the best performance with a time step of 100. <br>
Applied 5-fold cross-validation for accuracy assessment.

## 📊 Results
The proposed method significantly improves PDR accuracy across different motions compared to traditional step detection methods.

## 📂 Repository Structure
data/: Collected datasets for various motions. <br> 
models/: Pre-trained LSTM models and training scripts. <br>
notebooks/: Jupyter notebooks for data analysis and visualization. <br>
src/: Source code for feature preprocessing, model training, and evaluation. <br>
README.md: Project overview and usage instructions.

## 🚀 Getting Started
Prerequisites
Python 3.8 or later
TensorFlow, NumPy, Pandas, and other dependencies listed in requirements.txt
Installation

If you use this project in your research, please cite:
Kim, B., Enhancing Smartphone-Based Pedestrian Dead Reckoning Using AI in Various Motions, 2024 IPNT Conference, Nov 6-8 2024, Jeju, Korea

## 📧 Contact
For any questions or suggestions, please contact kimbosung1217@gmail.com

## 🤝 Contributing
We welcome contributions! Please check the contributing guidelines for more details.
