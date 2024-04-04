import streamlit as st
import ollama
st.title("LUNA: Virtual Assistant")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    res = ollama.chat(
        model="llava",
        messages=[
            {
                'role': 'user',
                'content': 'Describe this image:',
                'images': [uploaded_image]
            }
        ]
    )
    st.write("Description:", res['message']['content'])