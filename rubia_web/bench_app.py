
##############################################################
#
# Carregamento das bibliotecas, arquivos e afins
#
##############################################################


import base64 

import pandas as pd
import numpy as np
import streamlit as st

from session_state import get_state # session_state.py nesta pasta


# imagens

img_logoh = 'imagens/logo.png'
img_logov = 'imagens/logo.jpg'
img_top = 'imagens/sobre.png'
img_back_l = 'imagens/caminhao.jpg'
img_back_p = 'imagens/caminhaop.jpg'


# arquivos

url_uos = 'dados/dataset/uo_frota.csv'
url_analises = 'dados/dataset/analises_tratado.csv'
url_vigencias = 'dados/dataset/vigencias_consolidadas_tratado.csv'
url_eventos_acel = 'dados/dataset/eventos_diarios_acel_freada.csv'
url_eventos_velocidade = 'dados/dataset/eventos_diarios_velocidade.csv'
url_eventos_rpm = 'dados/dataset/eventos_diarios_rpm.csv'


# carregamento dos dados para o cache
@st.cache
def carregaDados(url):
    return pd.read_csv(url)

uo_frota = carregaDados(url_uos)



##############################################################
#
# Definição de funções locais para o contexto Veltec
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


# definição de uma classe para armazenar o estado atual de variáveis chave

class MyState: 

    def __init__(self, run_counter, screen, controls, uos):
        self.run_counter = run_counter
        self.screen = screen
        self.controls = controls
        self.uos = uos

def setup(run_counter, screen, controls, uos) -> MyState:
    print('Running setup')
    return MyState(run_counter, screen, controls, uos)

state = get_state(setup, run_counter=1, screen='home', controls=True, uos=[])




##############################################################
#
# Estilização CSS
#
##############################################################


css = f"""
<link href='https://fonts.googleapis.com/icon?family=Material+Icons' rel='stylesheet'>

<style>
    
    /* reset a few global parameters */
    html{{font-size: 1.0em;font-color:#262730 !important;}}
    @media only screen and (max-width: 400px){{
        html{{font-size: 1.0em;}}
    }}
    footer{{display: none;}}
    h1{{padding: 0px !important;}}
    hr{{margin: 0.5rem !important;}}
    label, p:not(.alert){{
        background-color: white !important;
        padding: 5px 15px;
        border-radius: 10px 20px !important;
        border: 1px solid #ccc;    
        box-shadow: 0 1px 4px rgba(0, 0, 0, .6);
    }}

    /* set the overall appearance and background for this app */
    .toolbar, .sidebar, .instructions{{
        visibility: hidden;
        display: none;
    }}
    .stApp{{
        background-color: white;
        opacity: 0.8;
    }}
    @media all and (orientation: portrait) {{ 
        .stApp:after{{
            content:'';
            background: url({encodeImg(img_back_p)}) no-repeat center center;
            background-size: cover;
            position: absolute;
            top:0px;
            left: 0px;
            width:100%;
            height:100%;
            z-index:-1;
            opacity: 0.5;
        }}
    }}
    @media all and (orientation: landscape) {{ 
        .stApp:after{{
            content:'';
            background: url({encodeImg(img_back_l)}) no-repeat center center;
            background-size: cover;
            position: absolute;
            top:0px;
            left: 0px;
            width:100%;
            height:100%;
            z-index:-1;
            opacity: 0.5;
        }}
    }}
    
    /* lets adjust our main containers positioning */
    .reportview-container .main .block-container{{
        max-width: 2000px;
        padding: 0.8rem;
        width: 100% !important;
        //border: 2px solid blue;
    }}
    div.block-container > div{{
        width: 100% !important;
        height: 100%;
        display: flex;
        flex-wrap: wrap;
        justify-content: space-evenly;
        //border: 1px solid red;
    }}
    div.element-container{{
        display: inline-block !important;
        height: auto !important;
        width: max-content !important;
        font-size: 1.2rem;    
        //border: 1px solid green;    
    }}
    .spacediv{{
        min-height: 3rem;
    }}

    /* now we can deep in our widgets setup */
    div.element-container > div.Widget:not(.stNumberInput):not(.stTimeInput):not(.stDateInput):not(.stTextInput):not(.stTextArea):not(.stSelectbox):not(.stMultiSelect):not(.stSlider){{
        width: max-content !important;
    }}
    .Widget > label{{
        font-size: 1rem;
    }}
    .Widget:not(.stRadio):not(.stSlider) > div{{
        box-shadow: 0 1px 4px rgba(0, 0, 0, .6);
    }}
    .alert{{
        margin: 0px;
        padding: 0.5rem;
    }}
    span.as{{background-color: #4248f5 !important;}}
    .stRadio label>div:first-child{{background-color: #4248f5 !important;}}
    .stSlider>div>div>div>div{{background-color: #4248f5 !important;}}
    .stSlider>div>div:first-child>div:first-child{{background: #aaa !important;}}
    .stMultiSelect span>span{{background-color: #4248f5 !important;}}
    .stSlider div{{color: #4248f5 !important;}}
    div.stEmpty{{
        height: 0px !important;
        padding: 0px !important;
        margin: 0px !important;
        width: 100% !important;
    }}
    div.element-container > div.stEmpty{{
        width: 100vh !important;
        display: block !important;
        visibility: hidden;
    }}

    /* here were gonna change our buttons layout to make them more beautifull and intuitive */
    button{{
        background-color: #4248f5 !important;
        color: white !important;
        border-color: white !important;
    }}
    button:hover{{
        background-color: blue !important;
    }}
    button:not(.sidebar-close):not(.control):not(.sidebar-collapse-control):not(.btn):not(.dropdown-item):not(.overlayBtn):not(.close){{
        padding: 5px 10px !important;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, .6);
        margin: 0rem 0.1rem !important;
    }}
    .sidebar button:not(.sidebar-close):not(.control):not(.sidebar-collapse-control):not(.btn):not(.dropdown-item):not(.overlayBtn):not(.close){{
        //width: 5rem;
        font-size: 1.0em;
    }}
    .main button:not(.sidebar-close):not(.control):not(.sidebar-collapse-control):not(.btn):not(.dropdown-item):not(.overlayBtn):not(.close){{
        //width: 2rem;
        font-size: 1.0em;
    }}
</style>"""
st.markdown(css, unsafe_allow_html=True) 

css_mostra = """<style>.toolbar, .sidebar, .instructions{ visibility: visible;display: block;}</style>"""   
css_esconde = """<style>.toolbar, .sidebar, .instructions{ visibility: hidden;display: none;}</style>"""   

placeholder_controles = st.empty()

state.controls = placeholder_controles.checkbox('Mostrar controles', True)
if state.controls:
    st.markdown(css_mostra, unsafe_allow_html=True) 
else:
    st.markdown(css_esconde, unsafe_allow_html=True) 




##############################################################
#
# Inicialização do APP e filtros de dados
#
##############################################################



st.title("VELTEC - Benchmark de Risco") 

max_frota = int(uo_frota["frota"].max())
minr, maxr = st.sidebar.slider("Tamanho de frota: ",0,max_frota,[0,max_frota])
data = uo_frota.loc[(uo_frota["frota"]>=minr)&(uo_frota["frota"]<=maxr)]
#st.dataframe(data.drop("uo_id_pai",axis=1))

data_uos = list(data["uo_id"])
state.uos = st.sidebar.multiselect("Selecionar UOs: ", data_uos, state.uos)

type_analysis = st.sidebar.radio("Benchmark:",("Segurança","Economia"))











st.button('Teste')
st.button('😃')
st.selectbox('Selecione', ['A','B'])
st.multiselect('Selecione', ['A','B'])
st.checkbox('Check')
st.radio('Radio', ['S','N'])
st.text_input('Nome')
st.number_input('Nome')
st.date_input('Nome')
st.time_input('Nome')
st.text_area('Nome')
st.markdown('---')
st.title('Título')
st.slider('Slider')