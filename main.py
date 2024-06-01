import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D
import json
from transformers import pipeline
import numpy as np
import re
# Define custom objects for model loading
custom_objects = {'Conv2D': Conv2D, 'Adam': tf.keras.optimizers.Adam}

# Tensorflow Model Prediction
def model_prediction(test_image):
    # Load the model without compiling
    model = load_model("trained_plant_disease_model.keras", custom_objects=custom_objects, compile=False)
    # Recompile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    
    # Get the index of the predicted class
    class_index = np.argmax(predictions)
    
    # Map class name to disease name
    class_name = class_names[class_index]
# Remove special characters like _ and ___
    disease_name = re.sub(r'[_]+', ' ', class_name).title()  # Replace one or more underscores with a space and capitalize

    return disease_name

# Define class names
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
               'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
               'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
               'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
               'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
               'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']

# Load the JSON data from a file
with open('disease_data.json', 'r') as file:
    json_data = json.load(file)

# Initialize the question-answering model
qa_model = pipeline("question-answering")

# Function to get advice for a specific disease
def get_advice(disease_name):
    for key, value in json_data.items():
        if value["disease_name"].lower() == disease_name.lower():
            return value["advice"]
    return "Disease not found."

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Disease Recognition"])



if  app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    
    if test_image and st.button("Predict"):
        with st.spinner("Predicting..."):
            predicted_disease = model_prediction(test_image)
            st.success(f"The disease name is {predicted_disease}")
            st.session_state['predicted_disease'] = predicted_disease

    if 'predicted_disease' in st.session_state:
        st.write("You can now ask questions about the disease advice.")

        if 'show_solution' not in st.session_state:
            st.session_state['show_solution'] = False
        
        if st.button("Ask Question "):
            st.session_state['show_solution'] = True
        
        if st.session_state['show_solution']:
            st.write("Enter your question below:")

            if 'user_question' not in st.session_state:
                st.session_state['user_question'] = ""
            if 'qa_response' not in st.session_state:
                st.session_state['qa_response'] = ""

            user_question = st.text_area("Ask a question about the disease advice:", value=st.session_state['user_question'])
            
            if st.button("Submit"):
                st.session_state['user_question'] = user_question
                if user_question:
                    with st.spinner("Answering..."):
                        disease_advice = get_advice(st.session_state['predicted_disease'])
                        qa_response = qa_model(question=user_question, context=disease_advice)
                        st.session_state['qa_response'] = qa_response['answer']
                        st.session_state['user_question'] = user_question
            
            if st.session_state['qa_response']:
                st.write(f"Answer: {st.session_state['qa_response']}")
