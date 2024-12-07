import streamlit as st
from PIL import Image
from modules.data_processing import *
from modules.matcher import *
import pickle
import cv2
import numpy as np

# Load objects
with open('models/xgb.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

train_images = np.load('tensors/train_images.npy', allow_pickle=True)
train_labels = np.load('tensors/train_labels.npy', allow_pickle=True)

# Define the app logic
if "page" not in st.session_state:
    st.session_state.page = "upload"  # Default page is the upload page

if st.session_state.page == "upload":
    # Upload Page
    st.title("Image Viewer")

    # Create a file uploader
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "bmp", "gif"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Submit"):
            
            with st.spinner("Processing image..."):
                # Read the image in grayscale
                query_image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                query_image = cv2.resize(query_image, (256, 256))
                query_image = np.array(preprocess_image(query_image))
                
                train_features_combined = extract_combined_features([query_image])
                train_features = scaler.fit_transform(train_features_combined)
                xgb_predictions = model.predict(train_features)

                predicted_label = encoder.inverse_transform(xgb_predictions)

                # Perform classification and matching
                best_label, best_train_image, match_img = classify_and_plot_matches(
                    query_image, train_images, train_labels, upper_limit=1024
                )

            # Store results in session state and navigate to the results page
            st.session_state.best_label = best_label
            st.session_state.match_img = match_img
            st.session_state.predicted_label = predicted_label
            st.session_state.page = "result"
            st.rerun()

elif st.session_state.page == "result":
    # Results Page
    st.title("Results")

    # Display classification results
    st.write(f"**Predicted Label by AI:** {st.session_state.predicted_label[0]}")
    st.write(f"**Best Label:** {st.session_state.best_label}")
    st.image(st.session_state.match_img, caption="Matched Image", use_container_width=True)

    # Option to go back to the upload page
    if st.button("Back to Upload"):
        st.session_state.page = "upload"
        st.rerun()
