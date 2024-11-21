# Skin Disease Classification with Symptom Matching

This repository contains a web-based application that predicts skin diseases based on uploaded images and user-provided symptoms. The application uses a combination of Vision Transformer (ViT) for image classification and Sentence Transformer for symptom matching to provide accurate predictions and detailed information about the identified diseases.

## Features
- **Image Classification**: Uses Vision Transformer (ViT) to classify skin diseases from uploaded images.
- **Symptom Matching**: Matches user-provided symptoms with predefined symptom lists using Sentence Transformers for improved prediction accuracy.
- **Disease Information**: Provides descriptions, symptoms, causes, and treatment options for the predicted disease.
- **Streamlit Interface**: Easy-to-use web interface for uploading images and entering symptoms.

## Supported Diseases
- Acne and Rosacea
- Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions
- Psoriasis and related diseases
- Tinea Ringworm Candidiasis and other Fungal Infections
- Nail Fungus and other Nail Diseases
- Seborrheic Keratoses
- Warts Molluscum

## Getting Started

### Prerequisites
- Python 3.7 or higher
- Required libraries:
  - `streamlit`
  - `transformers`
  - `sentence-transformers`
  - `torch`
  - `opencv-python`
  - `numpy`
  - `Pillow`

Install the required dependencies using:
```bash
pip install -r requirements.txt

Setup

    Clone this repository:

    git clone https://github.com/HumayounMustafa/SkinDiseaseViT.git
    cd SkinDiseaseViT

    Place the dat.json file containing disease information in the root directory.
    Download the pretrained weights for ViT and place them in the appropriate folder.

Run the Application

Start the Streamlit application:

streamlit run app.py

Upload Image and Symptoms

    Upload an image of the affected skin area.
    Enter symptoms separated by commas (e.g., itching, redness, dry skin).

The application will display:

    Predicted disease name
    Description
    Symptoms
    Causes
    Treatment options
    Match confidence score

Files in the Repository

    app.py: Streamlit application code.
    dat.json: JSON file containing disease information (descriptions, symptoms, causes, and treatments).
    ViTInference.ipynb: Notebook for inference using Vision Transformer.
    requirements.txt: List of dependencies.

Pretrained Weights

The application uses state-of-the-art pretrained weights for ViT. You can download them here.
Acknowledgments

    Hugging Face Transformers: For providing pretrained models and tools.
    Streamlit: For creating a simple and interactive web interface.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to contribute to this project by submitting issues or pull requests!
