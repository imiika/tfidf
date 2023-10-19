import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_option_menu import option_menu
import string # for removing punctuation
import nltk #natural language toolkit
from nltk import word_tokenize, download, stem #preprocessing
from nltk.corpus import stopwords, words #remove stopwords
import itertools

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

def read_data():
    doc_names = os.listdir( 'Documents/' )
    doc_names.reverse()
    docs = []
    for doc in doc_names :
        with open( 'Documents/'+doc, 'r' ) as file :
            docs.append( file.read() )
    return docs

def clean_preprocess(docs):
    stemmer = stem.PorterStemmer()
    preprocessed_docs = []
    for i in range( len(docs) ) :
        doc = docs[i].lower() # cleaning : lower case
        doc = ' '.join( doc.split() ) # cleaning: remove non words
        
        tokens = word_tokenize( doc ) # preprocessing : tokenization
        stop_words = nltk.corpus.stopwords.words( 'english' ) # preprocessing : stop words
        doc = ' '.join( [token for token in tokens if token not in stop_words] ) # preprocessing : stop words removal

        tokens = word_tokenize( doc ) # preprocessing : tokenization
        doc = ' '.join( [stemmer.stem(token) for token in tokens] ) # preprocessing : stemming
        preprocessed_docs.append( doc )
    return preprocessed_docs

def create_dicts(docs):
    dicts = []
    for i in range( len(docs) ) :
        words = list( docs[i].split(" ") )
        dictionary = {}
        for token in words :
            if token in dictionary.keys():
                dictionary[token] += 1
            else:
                dictionary[token] = 1
        
        dicts.append( dictionary )
    return dicts

def show_dicts(dict):
    words = [key for key, _ in dict.items()]
    freq = [value for _, value in dict.items()]

    dict_df = pd.DataFrame( {'Words':words, 'Frequencies':freq } )
    st.markdown( dict_df.style.hide( axis="index" ).to_html(), unsafe_allow_html = True )

def create_dict(docs):
    words = []
    distinct_words = []
    
    for i in range( len(docs) ) :
        tokens = list( docs[i].split(" ") )
        words.append( tokens )
        distinct_words.extend( tokens )
    
    distinct_words = list( set( distinct_words ) )

    doc_names = []
    for i in range( len(docs) ) :
        doc_names.append( "D"+str(i+1) )

    dictionary = { key : 0 for key in list( itertools.product( distinct_words, doc_names ) ) }
    for i in range( len(docs) ) :
        doc = 'D'+str(i+1)
        for word in words[i] :
            dictionary[ (word, doc) ] += 1

    return dictionary    

def show_dict( dict ):
    words = [key[0] for key, _ in dict.items()]
    docs = [key[1] for key, _ in dict.items()]
    freq = [value for _, value in dict.items()]

    dict_df = pd.DataFrame( {'Words':words, 'Documents':docs, 'Frequencies':freq } )
    st.markdown( dict_df.style.hide( axis="index" ).to_html(), unsafe_allow_html = True )
    
if choose == "Home" :
    st.title( "Reconnaissance des mots arabes manuscrits pris de la base de données IFN/ENIT" )

elif choose == "Dictionaries" :
    docs = read_data()
    preprocessed_docs = clean_preprocess(docs)

    for i in range( len(docs) ) :
        st.write( "## Text n", i+1 )
        st.write( "### Original text :" )
        st.write( docs[i] )
        
        st.write( "### Text after cleaning and preprocessing : lower case, stopwords and non-words removal, stemming :" )
        st.write( preprocessed_docs[i] )

        dicts = create_dicts( preprocessed_docs )
        st.write( "### Dictionary :" )
        show_dicts( dicts[i] )

elif choose == "TF-IDF" :
    docs = read_data()
    preprocessed_docs = clean_preprocess(docs)
    
    st.write( "## Original texts :" )
    for i in range( len(docs) ) :
        st.write( docs[i] )

    st.write( "## Preprocessed texts :" )
    for i in range( len(docs) ) :
        st.write( preprocessed_docs[i] )

    dict = create_dict( preprocessed_docs )
    st.write( "## Dictionary :" )
    show_dict( dict )
        
