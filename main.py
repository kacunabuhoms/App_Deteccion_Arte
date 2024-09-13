import streamlit as st

st.title("Deteccion de Arte")

image = st.capture_image("Ingrese imagen")


if image:
    st.success("Imagen capturada")
    st.image(image)
