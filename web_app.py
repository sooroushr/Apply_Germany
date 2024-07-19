import time
import numpy as np
import pandas as pd
import streamlit as st
import Apply_Extraction

st.title("University Program Search")
st.write("Enter the university and program you are interested in to search for relevant information")

AE = Apply_Extraction.Apply_Extraction()

university = st.text_input("University")
program = st.text_input("Program")




if st.button("Search For Information"):
    AE.Extract_Aspect(university, program)
    st.write('Rag Built')
    
    
    aspects = ['duration', 'teaching language', 'required german language level', 'required english language level', 'application deadline date', 'minimum required grade']
    
    
    for aspect in aspects:
        st.write(aspect)
        st.write( AE.Aspect_extract(aspect) )

