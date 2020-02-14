
##############################################################
#
# Package and resources loading
#
##############################################################


import base64 

import pandas as pd
import numpy as np
import streamlit as st

import scipy
from scipy.io.arff import loadarff

import SessionState
import rubia_models


# assets

img_logow = 'assets/images/logo_wide.png'
img_logos = 'assets/images/logo_small.png'

img_bg_portrait = 'assets/images/bg_portrait.jpg'
img_bg_landscape = 'assets/images/bg_landscape.jpg'

css = 'assets/styles/styles.css'


# session management

state = SessionState.get(screen='home', admin=True, file='', rm=None, success=False)





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


def loadCSS():
    print('CSS', np.random.random())
    style = """<style>%s</style>""" % loadTxt(css)
    style_MD = """<style><link href='https://fonts.googleapis.com/icon?family=Material+Icons' rel='stylesheet'></style>"""
    
    style = style.replace('LOGO_WIDE', encodeImg(img_logow))
    if not state.rm:
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
# Page layout management - SIDEBAR
#
##############################################################

# get data from the user selected file below
def loadData(uploaded_file):
    df = pd.DataFrame()
    if uploaded_file is not None and uploaded_file.getvalue() != state.file:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            try:
                df = pd.read_excel(uploaded_file)
            except:
                try:
                    uploaded_file.seek(0)
                    data, meta = scipy.io.arff.loadarff(uploaded_file)
                    df = pd.DataFrame(data)
                except:
                    st.error('File type not supported')
    if len(df) > 0:
        state.file = uploaded_file.getvalue()
        state.rm = rubia_models.rubia_models(df, width=120, debug=False)
    return df


st.sidebar.title('Setup')

st.sidebar.markdown('---')
ph_s1 = st.sidebar.empty()
ph_s2 = st.sidebar.empty()
ph_s3 = st.sidebar.empty()
ph_s31 = st.sidebar.empty()
ph_s4 = st.sidebar.empty()
ph_s5 = st.sidebar.empty()
ph_s6 = st.sidebar.empty()
ph_s7 = st.sidebar.empty()
st.sidebar.markdown('---')

# little hack to have the File Uploader styling - this class runs differently on streamlit 0.55
st.sidebar.markdown('<div class="spacediv"></div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Choose a CSV, XLSX or ARFF file", type=['csv', 'xlsx', 'arff'])
df = loadData(uploaded_file)
st.sidebar.markdown('<div class="spacediv"></div>', unsafe_allow_html=True)

# show data segmentation filters
if state.rm: 
    y_cols = [ph_s1.selectbox('Target column (y)', ['Unsupervised'] + list(state.rm.data_raw.columns))]
    if y_cols == ['Unsupervised']: y_cols = []
    ignore_cols = ph_s2.multiselect('Ignore features (xs)', state.rm.data_raw.columns)
    set_order = ph_s3.slider('Model of order', 1, 3, 2)
    set_trainingsize = ph_s4.slider('Test size', 0., 1., 0.3) 
    xtransform = ph_s5.selectbox('Transformation on X', ['None','Standard','MinMax','Robust'])
    ytransform = ph_s6.selectbox('Transformation on y', ['None','BoxCox'])
    balance_tol = ph_s7.slider('Auto balance tolerance', 0., 1., 0.3)


##############################################################
#
# Page layout management - MAIN
#
##############################################################

ph_c1 = st.empty()
ph_c2 = st.empty()
ph_c3 = st.empty()
ph_c4 = st.empty()
ph_c41 = st.empty()
ph_c5 = st.empty()
ph_c51 = st.empty()
ph_c52 = st.empty()
ph_c6 = st.empty()
ph_c7 = st.empty()
st.markdown('---')

if not state.rm:

    st.title('Rubia Models')
    st.warning('Please select a file using the left sidebar menu')


else:

    if ph_c1.checkbox('Describe RAW'):
        state.rm.describe(state.rm.data_raw, printt=False)
        st.title('DATA OVERVIEW')
        st.write('DATA SHAPE: ', state.rm.data_raw.shape)
        st.write('COLUMNS INFO: ', ', '.join(state.rm.cols_dtypes))
        st.write('DATA SAMPLE: ')
        st.write(state.rm.data_raw.sample(5))
        st.write('STATISTICS: ')
        st.write(state.rm.data_raw.describe(include='all').T)

    graph = ph_c2.checkbox('Exploratory graphs')  
    state.rm.explore(state.rm.data_raw, y_cols, ignore_cols, printt=False, graph=graph) #updates X, y, M and remove constant columns
    y_cols = list(state.rm.y.columns) # update the list of y columns after an EDA

    if graph: # show graphs for exploratory analysis
        for fig in state.rm.graphs_expl:
            st.pyplot(fig)     

    if ph_c4.checkbox('Encode', True): # encode non numeric or numeric like columns
        if ph_c41.checkbox('One Hot Encoder'):
            state.rm.encode(encoder='OneHotEncoder')
            y_cols = list(state.rm.y.columns) # update the list of y columns after a One Hot Encode process
        else:
            state.rm.encode(encoder='LabelEncoder')

    # add auto balance correction if applicable/selected
    if balance_tol > 0 and len(y_cols) == 1:
        state.rm.balance(balance_tol, state.rm.M, y_cols, ignore_cols)
    else:
        ph_s7.empty()

    if ph_c5.checkbox('Non linear'): # add more complex terms (non linear transformations of X)
        add_inter = ph_c51.checkbox('Add interaction')
        add_root = ph_c52.checkbox('Add root')
        state.rm.addTerms(state.rm.X, state.rm.y, levels=set_order, interaction=add_inter, root=add_root)
    
    state.rm.explore(state.rm.M, y_cols, ignore_cols, printt=False, graph=graph) #updates X, y, M and remove constant columns
    y_cols = list(state.rm.y.columns) # update the list of y columns after an EDA

    ph_b1 = st.empty()
    ph_b2 = st.empty()
    
    state.rm.analyse(y_cols) # decide the model type (regr, class, cluster)
    # only apply y transformation for single variable regression
    if state.rm.strategy != 'regression' or len(y_cols) > 1:
        ph_s6.empty()
        ytransform = 'None'

    countk = len(state.rm.X.columns)
    kbest = ph_s31.slider('Limit K best features', 1, countk, 10 if countk > 10 else countk)
    if kbest != len(state.rm.X.columns): state.rm.redux(k=kbest)
    
    if ph_c3.checkbox('Show features'):  
        st.title('FEATURE EXTRACTION REPORT')
        st.write('X: ', ' | '.join(state.rm.X.columns))
        st.write('y: ', ' | '.join(state.rm.y.columns)) 
        st.write('M: ', (state.rm.X.shape), '|', state.rm.y.shape)

    graphm = ph_c6.checkbox('Modeling graphs')

    if ph_b1.button('EVALUATE'): # start the evaluation and performance tests
        alphas = 10 ** np.linspace(10, -2, 100) * 0.5
        state.rm.evaluate(test_size=set_trainingsize, transformX=xtransform, transformY=ytransform, folds=10, alphas=alphas, printt=False, graph=graphm)
        st.title('RESULTS - BEFORE BOOSTING')
        st.write(state.rm.report_performance)

        # boost the best model
        best = state.rm.report_performance.Model.iloc[0]
        st.success(best)
        state.success = True
        if graphm: # show graphs for model evaluation and overall performance
            for fig in state.rm.graphs_model:
                st.pyplot(fig)     

    if state.success: # show advanced boosting parameters
        st.title('BOOSTING')
        boost = st.selectbox('Choose a model to boost', state.rm.report_performance.Model)
        st.success('Boosting model: ' + str(boost))
        if st.button('Choose & Boost'):
            result = state.rm.test(str(boost), printt=False, graph=graphm)
            st.success(result)
            if graphm: # show graphs for model evaluation and overall performance
                for fig in state.rm.graphs_model:
                    st.pyplot(fig)     

    if not state.success:
        st.title('AUTO MODELING - ' + state.rm.strategy.upper())

    if ph_c7.checkbox('Show data sample'):
        st.title('POST-PROCESSING DATA SAMPLE (X|y)')
        st.write(state.rm.X.head(5))
        st.write(state.rm.y.head(5)) 

    if ph_b2.button('CLEAR ALL'):
        state.rm = None
        state.success = False
        #ph_b1.empty()

