import cv2
import streamlit as st
from PIL import Image
import numpy as np
import os
from deepface import DeepFace

import tempfile
import tensorflow as tf
import json
import base64

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

SAVE_DIR = "D:\\registered_images"
REGISTERED_FACES_FILE = os.path.join(SAVE_DIR, "registered_faces.json")

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

if not os.path.exists(REGISTERED_FACES_FILE):
    with open(REGISTERED_FACES_FILE, "w") as f:
        json.dump([], f)

def load_registered_faces():
    if os.path.exists(REGISTERED_FACES_FILE):
        with open(REGISTERED_FACES_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_registered_faces(registered_faces):
    with open(REGISTERED_FACES_FILE, "w") as f:
        json.dump(registered_faces, f)

def get_base64(bin_file):
    with open(bin_file, 'rb') as file:
        return base64.b64encode(file.read()).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    background_css = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(background_css, unsafe_allow_html=True)

set_background("bg.jpg")

def set_logo(png_file):
    logo_str = get_base64(png_file)
    logo_css = f'''
    <style>
    .logo-container {{
        position: fixed;
        top: 0px;
        left: 0px;
        z-index: 1;
    }}
    .logo-container img {{
        width: 350px;
    }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_str}">
    </div>
    '''
    st.markdown(logo_css, unsafe_allow_html=True)

set_logo("streamlit_logo.png")

# # CSS for camera input placeholder
# st.markdown(
#     """
#     <style>
#     /* Style for camera input placeholder */
#     .css-19u5cly { 
#         color: white !important; 
#     }
#     /* Style adjustments for the select box */
#     div[data-baseweb="select"] > div {
#         background-color: chartreuse !important;
#     }
#     div[data-baseweb="select"] input {
#         color: white !important;
#     }
#     div[data-baseweb="select"] > div > span {
#         color: white !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# ) 

# Custom CSS for the selectbox style
st.markdown(
    """
    <style>
    /* Set all text color to white */
    .stApp, .stApp * {
        color: brown !important;
    }

    /* Custom style for the selectbox */
    div[data-baseweb="select"] > div {
        background-color: chartreuse !important; /* Chartreuse background color for selectbox */
    }
    div[data-baseweb="select"] input {
        color: brown !important; /* Black text inside the selectbox */
    }
    div[data-baseweb="select"] > div > span {
        color: brown !important; /* Black text for selected option */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Welcome to My Streamlit App")
st.write("This app has a custom background image!")

def main():
    if "registered_faces" not in st.session_state:
        st.session_state.registered_faces = load_registered_faces()

    action = st.selectbox("Choose option", ["Register", "Authenticate"])

    if action == "Register":
        st.write("Register your face in our database")

        enable = st.checkbox("Disable camera")
        picture = st.camera_input("Take a picture", disabled=enable)
        if picture:
            img_pil = Image.open(picture)
            st.image(img_pil, caption="Registered Image", use_column_width=True)

            img_name = f"registered_{len(st.session_state.registered_faces) + 1}.jpg"
            img_path = os.path.join(SAVE_DIR, img_name)
            img_pil.save(img_path)
            st.session_state.registered_faces.append(img_path)
            save_registered_faces(st.session_state.registered_faces)
            st.success(f"Image saved to {img_path}")

    elif action == "Authenticate":
        st.write("Face verification")

        if not st.session_state.registered_faces:
            st.warning("No registered faces found. Please register your face first.")
        else:
            enable = st.checkbox("Disable camera")
            picture = st.camera_input("Take a picture", disabled=enable)
            if picture:
                img_pil = Image.open(picture)
                st.image(img_pil, caption="Verification Photo", use_column_width=True)
                
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                    img_pil.save(tmp_file.name)
                    tmp_img_path = tmp_file.name
                
                verified = False
                for registered_face in st.session_state.registered_faces:
                    try:
                        verification_result = DeepFace.verify(
                            img1_path=registered_face,
                            img2_path=tmp_img_path,
                            model_name="VGG-Face"
                        )
                        if verification_result["verified"]:
                            st.success(f"Verification successful with image {registered_face}")
                            verified = True
                            break
                    except Exception as e:
                        st.error("Error: " + str(e))

                if not verified:
                    st.error("Not Authorized, Verification Unsuccessful !!!")

                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)

if __name__ == "__main__":
    main()
