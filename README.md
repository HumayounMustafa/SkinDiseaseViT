# Skin Disease Classification with Symptom Matching
## Overview

This repository contains a Streamlit application that predicts skin diseases based on uploaded images and user-provided symptoms. The application combines image classification using a fine-tuned Vision Transformer (ViT) model and symptom matching using a Sentence Transformer to enhance prediction accuracy.
## Table of Contents

    Features
    Supported Skin Conditions
    Model Weights
    Installation
    Usage
    How It Works
    Data
    Files in the Repository
    Contributing
    License
    Acknowledgments
    Contact

### Features

    Image Classification: Classifies skin diseases from images using a fine-tuned ViT model.
    Symptom Matching: Enhances predictions by matching user-input symptoms with predefined symptoms using semantic similarity.
    Interactive Interface: Built with Streamlit for an easy and interactive user experience.

### Supported Skin Conditions

    Acne/Rosacea
    Actinic Keratosis Basal Cell Carcinoma
    Psoriasis and Lichen Planus
    Tinea Ringworm Candidiasis
    Nail Fungus and other Nail Diseases
    Seborrheic Keratoses
    Warts Molluscum

### Model Weights

Our fine-tuned ViT model achieves state-of-the-art performance on skin disease classification tasks on Hugging Face. The trained weights are available at https://drive.google.com/drive/folders/1b_IO8jqYP4oMPpBanhQvhEPdXMtLNUDE?usp=sharing. Please download the weights and place them in a directory named ViTModel in the root of the repository.
## Installation

### Clone the repository

git clone https://github.com/HumayounMustafa/SkinDiseaseViT.git
cd SkinDiseaseViT

### Create a virtual environment

python3 -m venv venv
source venv/bin/activate

### Install the required packages

    pip install -r requirements.txt

### Requirements:
        streamlit
        transformers
        sentence_transformers
        torch
        torchvision
        numpy
        opencv-python
        Pillow

### Download the pre-trained models
        ViT Model: Download from https://drive.google.com/drive/folders/1b_IO8jqYP4oMPpBanhQvhEPdXMtLNUDE?usp=sharing and place the weights in a directory named ViTModel.
        Sentence Transformer Model: The code uses all-MiniLM-L6-v2, which will be automatically downloaded.

### Ensure dat.json is present

    The dat.json file contains detailed information about each skin disease, including descriptions, symptoms, causes, and treatments. Ensure that this file is present in the root directory.

### Usage

    Run the Streamlit app

    streamlit run app.py

    Interact with the app
        Enter Symptoms: Input your symptoms separated by commas (e.g., itching, redness, dry skin).
        Upload Image: Upload a clear image of the affected skin area.
        Get Prediction: The app will display the predicted disease along with detailed information.

### How It Works

    Image Classification
        The uploaded image is converted from BGR to RGB and then to a PIL image.
        The image is preprocessed using ViTImageProcessor.
        The ViT model predicts the skin disease based on the image.

    Symptom Matching
        User-input symptoms are encoded using a Sentence Transformer model.
        Predefined symptoms for each disease are also encoded.
        Cosine similarity is calculated between user symptoms and predefined symptoms.
        The disease with the highest similarity score (above a threshold) is selected.

    Final Prediction
        Combines image classification and symptom similarity.
        If the predicted disease matches the highest symptom similarity and the similarity score is above 0.7, detailed information is displayed.
        If another disease has a higher similarity score, possible diseases are suggested.
        If no sufficient match is found, the user is prompted to provide more symptoms.

### Data

The dat.json file contains information about each supported skin condition, including:

    Description: A brief overview of the disease.
    Symptoms: Common symptoms associated with the disease.
    Causes: Possible causes or risk factors.
    Treatment: Links to resources for treatment options.
    Symptom List: A list of symptoms used for similarity matching.

### Sample dat.json content:

{
  "Acne and Rosacea": {
    "description": "Acne is a long-term skin disease...",
    "symptoms": "Scars and Pigmentation",
    "causes": "Risk factors include hormones, infections...",
    "treatement-1": "https://www.medicinenet.com/acne/article.htm",
    "symptom_list": "redness, pimples, pustules, swelling..."
  },
  ...
}

### Files in the Repository

    app.py: The main Streamlit application.
    dat.json: Contains disease information.
    ViTInference.ipynb: Jupyter notebook for model inference demonstration.
    requirements.txt: Python package requirements.

### Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.
License

This project is licensed under the MIT License.
### Acknowledgments

    Hugging Face Transformers: For the ViT and Sentence Transformer models.
    Streamlit: For providing an easy way to build web apps.
    OpenCV: For image processing.

Contact

For questions or support, please contact [your email].
