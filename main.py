import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_option_menu import option_menu

with st.sidebar :
    choose = option_menu("Menu", ["Home", "Dictionaries", "TF-IDF", "Contact", "Help"],
                         icons=['house', 'table-fill', 'table', 'stars','person lines fill'],
                         menu_icon="menu-app",
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
