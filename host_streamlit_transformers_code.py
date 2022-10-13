"""
Streamlit App for Pest Disease Detection Trained using Huggingface Transformers
"""


# Imports
import torch
import streamlit as st
from PIL import Image
from transformers import AutoModelForImageClassification, AutoFeatureExtractor


# Classes
class Predict:
    def __init__(self, model_path="./results"):
        tb_path = 'C:\Users\model_path\'
        tbs_path = 'C:\Users\model_path\'
        tdh_path = 'C:\Users\model_path\'
        tfs_path = 'C:\Users\model_path\'
        tlf_path = 'C:\Users\model_path\'
        twe_path = 'C:\Users\model_path\'
        
        
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.img = self.get_image_from_upload()
        if self.img is not None:
            self.display_output()
            self.get_prediction()

    @staticmethod
    def get_image_from_upload():
        uploaded_file = st.file_uploader("Upload Files", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            return Image.open(uploaded_file)
        return None

    def display_output(self):
        st.image(self.img, caption='Uploaded Image')

    def get_prediction(self):
        if st.button('Classify'):
            with torch.no_grad():
                encoding = self.feature_extractor(self.img.convert("RGB"), return_tensors="pt")
                outputs = self.model(**encoding)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_idx = logits.argmax(-1).item()
            st.write(f'**Prediction**: {self.model.config.id2label[predicted_class_idx]}')
            st.write(f'**Probability**: {probs[0][predicted_class_idx].item()*100:.02f}%')
        else:
             st.write(f'Click the button to classify')


# Main
if __name__ == '__main__':
    st.title('Pest Disease Detection')
    st.subheader('Upload an image of a plant to classify the disease')
    predictor = Predict()
