import streamlit as st


try:
    model_name = st.session_state.model_name
    model = st.session_state.model
    st.markdown(f"<h1 style='text-align: center;'>You are using {model_name}!</h1>", unsafe_allow_html=True)
    with st.expander(f"{model_name} Summary"):
        model.summary(print_fn=st.write,expand_nested=True)
    
except:
    st.write("No model found!")