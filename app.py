import streamlit as st

st.set_page_config(
    page_title="Banana Disease Detection",
    layout="centered"
)

st.title("🍌 Banana Disease Detection")

st.success("Streamlit Deployment Successful!")

uploaded_file = st.file_uploader(
    "Upload Banana Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")
