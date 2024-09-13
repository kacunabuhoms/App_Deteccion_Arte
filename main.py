import streamlit as st

st.title("Deteccion de Arte")

image = st.camera_input("Capturar imagen")


if image:
    st.success("Imagen capturada")
    st.image(image)
