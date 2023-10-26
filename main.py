import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_option_menu import option_menu
import string # for removing punctuation
import nltk #natural language toolkit
from nltk import word_tokenize, download, stem, RegexpTokenizer #preprocessing
from nltk.corpus import stopwords, words #remove stopwords
import itertools

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

with st.sidebar :
    choose = option_menu("Menu", ["Home", "Dictionary per document", "Descriptors", "TF-IDF", "Contact", "Help"],
                         icons=['house', 'table 2 columns', 'search', 'file-earmark-binary', 'stars','person lines fill'],
                         menu_icon="menu-app",
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )

def read_data():
    doc_names = ['D1.txt', 'D2.txt', 'D3.txt', 'D4.txt', 'D5.txt', 'D6.txt']
    docs = []
    for doc in doc_names :
        with open( 'Documents/'+doc, 'r' ) as file :
            docs.append( ' '.join( file.read().split() ) )
    return docs

def clean_preprocess( docs, tokenization_method, stemmer ) :
    new_docs = docs.copy()
    if( stemmer == 'porter' ) :
        stemmer = stem.PorterStemmer()
    elif( stemmer == 'lancaster' ) :
        stemmer = stem.LancasterStemmer()
        
    for i in range( len(new_docs) ) :
        new_docs[i] = new_docs[i].lower() # cleaning : lower case
        
        if( tokenization_method == 'tokenize' ) :
            ExpReg = RegexpTokenizer('(?:[A-Z]\.)+|\d+(?:\.\d+)?DA?|\w+|\.{3}') 
            tokens = ExpReg.tokenize( new_docs[i] ) # preprocessing : tokenization
        elif( tokenization_method == 'split' ) :
            tokens = new_docs[i].split()

        stop_words = nltk.corpus.stopwords.words( 'english' ) # preprocessing : stop words
        tokens = [token for token in tokens if token not in stop_words]# preprocessing : stop words removal
            
        new_docs[i] = ' '.join( [stemmer.stem(token) for token in tokens] ) # preprocessing : stemming
    return new_docs

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

    dict_df = pd.DataFrame( {'Words':words, 'Occurrences':freq } )
    st.markdown( dict_df.style.hide( axis="index" ).to_html(), unsafe_allow_html = True )

def create_dict(docs):
    words = []
    distinct_words = []
    
    for i in range( len(docs) ) :
        tokens = list( docs[i].split(" ") )
        words.append( tokens )
        distinct_words.extend( tokens )
    
    distinct_words = sorted( list( set( distinct_words ) ) )

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

    dict_df = pd.DataFrame( {'Words':words, 'Documents':docs, 'Occurrences':freq } )
    st.markdown( dict_df.style.hide( axis="index" ).to_html(), unsafe_allow_html = True )
    
if choose == "Home" :
    st.title( "Information Representation : Indexing & TF-IDF : Term Frequency–Inverse Document Frequency" )

elif choose == "Dictionary per document" :
    st.title( "Dictionary per document" )
    
    docs = read_data()
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    st.write("")
    st.write("")
    st.write("")

    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split', 'porter')
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split', 'lancaster')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'tokenize', 'porter')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'tokenize', 'lancaster')
    
    if( option != '-' and option1 != '-' ) :
        for i in range( len(docs) ) :
            st.write( "## Text n", i+1 )
            st.write( "### - Original text :" )
            st.write( "#####", docs[i] )
            
            st.write( "### - Text after cleaning and preprocessing : lower case, stopwords and non-words removal, stemming :" )
            st.write( "#####", preprocessed_docs[i] )
    
            dicts = create_dicts( preprocessed_docs )
            st.write( "### Dictionary :" )
            show_dicts( dicts[i] )        

elif choose == "TF-IDF" :
    st.title( "TF-IDF" )
    docs = read_data()
    
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    st.write("")
    st.write("")
    st.write("")

    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split', 'porter')
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split', 'lancaster')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'tokenize', 'porter')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'tokenize', 'lancaster')
    
    st.write( "## ⚬ Original texts :" )
    for i in range( len(docs) ) :
        st.write( docs[i] )

    st.write( "## ⚬ Preprocessed texts :" )
    for i in range( len(docs) ) :
        st.write( preprocessed_docs[i] )

    dict = create_dict( preprocessed_docs )
    st.write( "## Dictionary :" )
    show_dict( dict )
        
elif choose == "Contact" :
    st.write( "### We are always happy to hear from you!" )
    st.write( "### Send us an email and tell us how we can help you via this email: tfidf@gmail.com" )
    
elif choose == "Help" :
    st.write( "### - Dictionary : A Python data structure storing informations per index/value" )
    st.write( "### - TF-IDF : Term Frequency–Inverse Document Frequency" )
