import streamlit as st
import numpy as np
import pandas as pd
import os
from streamlit_option_menu import option_menu
import string # for removing punctuation
import nltk #natural language toolkit
from nltk import word_tokenize, download, stem, RegexpTokenizer #preprocessing
from nltk.corpus import stopwords, words #remove stopwords
import itertools, math

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('words')

with st.sidebar :
    choose = option_menu("Menu", ["Home", "Descriptors", "Inverse Documents", "Dictionary per document", "TF-IDF", "Contact", "Help"],
                         icons=['house', 'file-earmark-binary', 'search', 'table 2 columns', 'search', 'stars', 'person lines fill'],
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
    if( stemmer == 'Porter stemmer' ) :
        stemmer = stem.PorterStemmer()
    elif( stemmer == 'Lancaster stemmer' ) :
        stemmer = stem.LancasterStemmer()
        
    for i in range( len(new_docs) ) :
        new_docs[i] = new_docs[i].lower() # cleaning : lower case
        
        if( tokenization_method == 'nltk.RegexpTokenizer.tokenize()' ) :
            ExpReg = RegexpTokenizer('(?:[A-Za-z]\.)+|[A-Za-z]+[\-@]\d+(?:\.\d+)?|\d+[A-Za-z]+|\d+(?:[\.\,]\d+)?%?|\w+(?:[\-/]\w+)*') 
            tokens = ExpReg.tokenize( new_docs[i] ) # preprocessing : tokenization
        elif( tokenization_method == 'split()' ) :
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
    
def query( tfidf, tokens, tokenization_method, stemmer ) :
    tokens = clean_preprocess( tokens, tokenization_method, stemmer )
    answers = [ "{:<10} {:<10} {:<10} {:<10}".format( 'Word', 'Document', 'Frequency', 'Weight') ]
    for line in tfidf :
        items = line.split()
        if( items[0] in tokens ) :
            answers.append( "{:<10} {:<10} {:<10} {:<10}".format( items[0], items[1], items[2], items[3] ) )
    if len(answers) == 1 :
        answers = []
    return answers

def matching( tfidf, tokens, tokenization_method, stemmer, matching_method ) :
    tokens = clean_preprocess( tokens, tokenization_method, stemmer )
    answers = []
    for line in tfidf :
        items = line.split()
        if( items[0] in tokens ) :
            answers.append( (items[1], float(items[3])) )
    if len(answers) == 0 :
        return answers
        
    documents = list( set( list(zip(*answers))[0] ) )
    sum_ = []
    for document in documents :
        sum_.append( round(np.sum( [answers[i][1] for i in range(len(answers)) if answers[i][0]==document] ), 4) )
    
    if( matching_method == 'scalar product' ) :    
        answers = sorted(list(zip(documents, sum_)), key=lambda x: x[1], reverse = True)
        return ["{:<10} {:<10}".format( answers[i][0], answers[i][1] ) for i in range(len(answers)) ]
    
    elif matching_method == 'cosine measure' :
        cosine_measure = []
        for i in range(len(sum_)) :
            sum_v = len( [answers[j][0] for j in range(len(answers)) if answers[j][0]==documents[i]] )
            sum_w = np.sum( [(answers[j][1])**2 for j in range(len(answers)) if answers[j][0]==documents[i]] )
            cosine_measure.append( round(sum_[i] / ( math.sqrt(sum_v) * math.sqrt(sum_w) ), 4) )
        answers = sorted(list(zip(documents, cosine_measure)), key=lambda x: x[1], reverse = True)
        return ["{:<10} {:<10}".format( answers[i][0], answers[i][1] ) for i in range(len(answers)) ]
    
    elif matching_method == 'jaccard measure' :
        jaccard_measure = []
        for i in range(len(sum_)) :
            sum_v = len( [answers[j][1] for j in range(len(answers)) if answers[j][0]==documents[i]] )
            sum_w = np.sum( [(answers[j][1])**2 for j in range(len(answers)) if answers[j][0]==documents[i]] )
            jaccard_measure.append( round(sum_[i] / ( sum_v + sum_w - sum_[i] ), 4) )
        answers = sorted(list(zip(documents, jaccard_measure)), key=lambda x: x[1], reverse = True)
        return ["{:<10} {:<10}".format( answers[i][0], answers[i][1] ) for i in range(len(answers)) ]

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
        preprocessed_docs = clean_preprocess(docs, 'split()', 'Porter stemmer')
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split()', 'Lancaster stemmer')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'nltk.RegexpTokenizer.tokenize()', 'Porter stemmer')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'nltk.RegexpTokenizer.tokenize()', 'Lancaster stemmer')
    
    if( option != '-' and option1 != '-' ) :
        for i in range( len(docs) ) :
            st.write( "## Text n", i+1 )
            st.write( "### - Original text :" )
            st.write( "#####", docs[i] )
            
            st.write( "### - Text after cleaning and preprocessing : lower case, stopwords and non-words removal, ", option1.split(" ")[0], " stemming :" )
            st.write( "#####", preprocessed_docs[i] )
    
            dicts = create_dicts( preprocessed_docs )
            st.write( "### Dictionary :" )
            show_dicts( dicts[i] )        

elif choose == "Descriptors" :
    st.title( "Descriptors" )
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    st.write("")
    st.write("")
    st.write("")   
    
    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        desc = open( "Files/descriptor_split_porter.txt", 'r' )
        
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        desc = open( "Files/descriptor_split_porter.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        desc = open( "Files/descriptor_split_porter.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        desc = open( "Files/descriptor_tokenize_lancaster.txt", 'r' )

    if( option != '-' and option1 != '-' ) :
        st.write( "## ⚬ Descriptor :" )
        for line in desc :
            st.text( line )
            
elif choose == "Inverse Documents" :
    st.title( "Inverse Documents" )
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    option2 = st.selectbox( "Choose what do you want to search about : ", ('-', 'Informations about a specific query', 'Matching for a specific query') )
    
    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        inv = open( "Files/tfidf_split_porter.txt", 'r' )
        
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        inv = open( "Files/tfidf_split_lancaster.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        inv = open( "Files/tfidf_tokenize_porter.txt", 'r' )
        
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        inv = open( "Files/tfidf_split_porter.txt", 'r' )
            
    if( option2 == 'Matching for a specific query' ) :
        option3 = st.selectbox( "Choose the matching model : ", ('-', 'Vector space model', 'Probabilistic model (BM25)') )
    
        if( option3 == 'Vector space model' ) :
            option4 = st.selectbox( "Choose matching measure : ", ('-', 'scalar product', 'cosine measure', 'jaccard measure') )
    st.write("")
    st.write("")
    st.write("")   
    
    if( option2 == 'Informations about a specific query' or ( option2 == 'Matching for a specific query' and option3 == 'Probabilistic model (BM25)') or (option2 == 'Matching for a specific query' and option3 == 'Vector space model' and option4 != '-') ) :
        col1, col2 = st.columns(2)
        with col1 :
            tokens = st.text_input( "Query 👇" )
        with col2 :
            button_search = st.button( 'search', key='search' )
        
    if( option != '-' and option1 != '-' and option2 != '-' ) :
        if( option2 == 'Informations about a specific query' ) :
            if( button_search ) :
                tokens = [ token for token in tokens.split() ]
                st.write( "## ⚬ Query's result :" )
                results = query( inv, tokens, option, option1 )
                if not results :
                    st.write( "No results found for this (these) word(s)." )
                else :
                    for line in results :
                        st.text( line )
        elif( option2=='Matching for a specific query' and option3=='Vector space model' and option4 != '-' ) :
            if( button_search ) :
                tokens = [ token for token in tokens.split() ]
                st.write( "## ⚬ Query's result :" )
                results = matching( inv, tokens, option, option1, option4 )
                if not results :
                    st.write( "No results found for this (these) word(s)." )
                else :
                    st.write( "{:<10} {:<10}".format( 'Document', 'Relevance' ) )
                    for line in results :
                        st.text( line )
        
elif choose == "TF-IDF" :
    st.title( "TF-IDF" )
    docs = read_data()
    
    option = st.selectbox( "Choose the term extraction method : ", ('-', 'split()', 'nltk.RegexpTokenizer.tokenize()') )
    
    option1 = st.selectbox( "Choose the stemmer : ", ('-', 'Porter stemmer', 'Lancaster stemmer') )
    st.write("")
    st.write("")
    st.write("")

    if( option == 'split()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split()', 'Porter stemmer')
    if( option == 'split()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'split()', 'Lancaster stemmer')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Porter stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'nltk.RegexpTokenizer.tokenize()', 'Porter stemmer')
    if( option == 'nltk.RegexpTokenizer.tokenize()' and option1 == 'Lancaster stemmer' ) :
        preprocessed_docs = clean_preprocess(docs, 'nltk.RegexpTokenizer.tokenize()', 'Lancaster stemmer')

    if( option != '-' and option1 != '-' ) :
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
