import streamlit as st
import json
import cv2
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from PIL import Image

# Load the ViT model
model = ViTForImageClassification.from_pretrained('ViTModel')
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model.eval()  # Set model to evaluation mode

# Load the JSON data
with open('dat.json') as f:
    data = json.load(f)

keys = list(data)

# Load the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similarity(user_symptoms, predefined_symptoms):
    user_embedding = embedding_model.encode(user_symptoms, convert_to_tensor=True)
    predefined_embedding = embedding_model.encode(predefined_symptoms, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(user_embedding, predefined_embedding)
    
    # Average the similarity scores
    average_score = cosine_scores.mean().item()
    return average_score

def predict(image, user_symptoms):
    # Convert OpenCV image to PIL Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    pil_image = Image.fromarray(image)

    # Preprocess the image using ViTImageProcessor
    inputs = processor(pil_image, return_tensors="pt")

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predicted label index
    predicted_class_idx = logits.argmax(-1).item()

    # Get predicted disease name
    predicted_disease_name = model.config.id2label[predicted_class_idx]

    # Proceed as before
    predicted_symptoms = data[predicted_disease_name]['symptom_list']

    # Check similarity with the predicted disease
    predicted_similarity = get_similarity(user_symptoms, predicted_symptoms)

    # Check similarity with all other diseases
    best_match_disease = predicted_disease_name
    best_match_similarity = predicted_similarity

    for disease in keys:
        if disease != predicted_disease_name:
            other_symptoms = data[disease]['symptom_list']
            other_similarity = get_similarity(user_symptoms, other_symptoms)

            if other_similarity > best_match_similarity:
                best_match_disease = disease
                best_match_similarity = other_similarity

    # If the best match is the predicted disease and similarity is >= 70%
    if best_match_disease == predicted_disease_name and best_match_similarity >= 0.7:
        description = data[predicted_disease_name]['description']
        symptoms = data[predicted_disease_name]['symptoms']
        causes = data[predicted_disease_name]['causes']
        treatment = data[predicted_disease_name]['treatement-1']
        return predicted_disease_name, description, symptoms, causes, treatment, best_match_similarity

    # If another disease matches better
    elif best_match_similarity >= 0.7:
        return f"Possible diseases: {predicted_disease_name} or {best_match_disease}", "", "", "", "", best_match_similarity

    # If no sufficient match is found
    else:
        return None, None, None, None, None, best_match_similarity

# Streamlit interface
st.title("Skin Disease Classification with Symptom Matching")
st.write("""
This application predicts skin diseases based on the image you provide.
It supports the following conditions:
- Acne and Rosacea
- Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions
- Psoriasis and related diseases
- Tinea Ringworm Candidiasis and other Fungal Infections
- Nail Fungus and other Nail Diseases
- Seborrheic Keratoses
- Warts Molluscum
""")

# User input for symptoms
user_symptoms = st.text_input("Enter your symptoms, separated by commas (e.g., itching, redness, dry skin)")

uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and user_symptoms:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)
    
    # Predict the disease
    disease_name, description, symptoms, causes, treatment, similarity = predict(image, user_symptoms)
    if disease_name=='Acne and Rosacea':
        disease_name='Acne/Rosacea'
    if disease_name:
        # Display the results
        st.subheader("Prediction Results")
        st.write(f"**Name of Disease:** {disease_name}")
        if description:  # Only show detailed info if a single disease is identified
            st.write(f"**Description:** {description}")
            st.write(f"**Symptoms:** {symptoms}")
            st.write(f"**Causes:** {causes}")
            st.write(f"**Treatment:** {treatment}")
        st.write(f"**Match Confidence:** {similarity:.2f}")
    else:
        st.write("Please provide more symptoms")
