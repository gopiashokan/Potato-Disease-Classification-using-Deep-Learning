import numpy as np
import PIL.Image as Image
import tensorflow as tf
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from warnings import filterwarnings
filterwarnings('ignore')


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Classification', layout='centered')

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Potato Disease Classification</h1>',
                unsafe_allow_html=True)
    add_vertical_space(4)


# Streamlit Configuration Setup
streamlit_config()


def prediction(image_path, class_names=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']):

    img = Image.open(image_path)
    img_resized = img.resize((256,256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    model = tf.keras.models.load_model(r'model\model.h5')
    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(np.max(prediction)*100, 2)

    add_vertical_space(1)
    st.markdown(f'<h4 style="color: orange;">Predicted Class : {predicted_class}<br>Confident : {confidence}%</h3>', 
                    unsafe_allow_html=True)
    
    add_vertical_space(1)
    st.image(img.resize((400,300)))


col1,col2,col3  = st.columns([0.1,0.9,0.1])
with col2:
    input_image = st.file_uploader(label='Upload the Image', type=['jpg', 'jpeg', 'png'])


if input_image is not None:

    col1,col2,col3 = st.columns([0.2,0.8,0.2])
    with col2:
        prediction(input_image)

