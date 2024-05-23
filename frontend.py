import streamlit as st
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('iris_model.h5')

# Streamlit interface
st.title('Iris Species Prediction')
st.write("Enter the parameters:")

sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, step=0.1)

# Make the prediction
if st.button('Predict'):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    st.write(f'Predicted Species: {iris.target_names[np.argmax(prediction)]}')