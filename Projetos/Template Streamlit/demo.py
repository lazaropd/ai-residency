
##############################################################
#
# Package and resources loading
#
##############################################################


import base64 

import pandas as pd
import numpy as np
import streamlit as st

import SessionState


# assets

img_logow = 'assets/images/logo_wide.png'
img_logos = 'assets/images/logo_small.png'

img_bg_portrait = 'assets/images/bg_portrait.jpg'
img_bg_landscape = 'assets/images/bg_landscape.jpg'

css = 'assets/styles/styles.css'


# session management

state = SessionState.get(screen='home', admin=False, file='')





##############################################################
#
# Local functions
#
##############################################################


def bytesTo64(bytes_file, header):
    encoded = base64.b64encode(bytes_file).decode()
    base64file = "data:%s;base64,%s" % (header, encoded)
    return base64file

def encodeImg(filename, filetype='image/jpeg'):
    fig = filename
    image = open(fig, 'rb').read()
    image64 = bytesTo64(image, filetype)
    return image64

def loadTxt(filename):
    with open(filename, 'r') as file:
        txt = file.read()
    return txt





##############################################################
#
# CSS styling
#
##############################################################

@st.cache
def loadCSS():
    print('CSS', np.random.random())
    style = """<style>%s</style>""" % loadTxt(css)
    style_MD = """<style><link href='https://fonts.googleapis.com/icon?family=Material+Icons' rel='stylesheet'></style>"""
    
    style = style.replace('LOGO_WIDE', encodeImg(img_logow))
    style = style.replace('BG_PORTRAIT', encodeImg(img_bg_portrait))
    style = style.replace('BG_LANDSCAPE', encodeImg(img_bg_landscape))

    return style + style_MD

style = loadCSS()
st.markdown(style, unsafe_allow_html=True) 

style_showcontrols = """<style>.toolbar, .instructions{ visibility: visible;display: block;}</style>"""   
style_hidecontrols = """<style>.toolbar, .instructions{ visibility: hidden;display: none;}</style>"""   
if state.admin:
    st.markdown(style_showcontrols, unsafe_allow_html=True) 
else:
    st.markdown(style_hidecontrols, unsafe_allow_html=True) 





##############################################################
#
# Page layout management
#
##############################################################


st.sidebar.multiselect('Multiselect', ['A','B','C','D','E'], ['A','E'])
st.sidebar.number_input('Number input', min_value=0., max_value=100., format='%.0f', step=1.)
st.sidebar.radio('Radio', ['S','N'])
st.sidebar.markdown('---')
if st.sidebar.text_input('Password', type="password") == '1234': state.admin = True
st.sidebar.button('🔐')

st.checkbox('Check')
st.checkbox('Another')
st.checkbox('Yet')
st.markdown('---')
st.title('Título')
st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.')
st.text_input('Text input')
st.date_input('Date input')
st.text_area('Textarea')
st.slider('Slider', 0, 100, (30, 55))
st.success('Lorem ipsum')
st.error('Lorem ipsum')
st.button('BUTTON')
st.button('😃')
st.button('ANOTHER')
st.markdown('---')

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

