import streamlit as st
import numpy as np
import os
from streamlit_option_menu import option_menu
import string # for removing punctuation
import nltk #natural language toolkit
from nltk import word_tokenize, download, stem #preprocessing
from nltk.corpus import stopwords, words #remove stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

with st.sidebar :
    choose = option_menu("Menu", ["Home", "Dictionaries", "TF-IDF", "Contact", "Help"],
                         icons=['house', 'table 2 columns', 'table', 'stars','person lines fill'],
                         menu_icon="menu-app",
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

def read_data :
    doc_names = os.listdir( 'Documents/' )
    docs = []
    for doc in doc_names :
        with open( 'Documents/'+doc, 'r' ) as file :
            docs.append( file.read() )
        
    # Print data :
    for doc in docs :
        print(doc, '\n')
