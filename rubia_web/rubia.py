import os
import math 
import textdistance
import numpy as np
import pandas as pd

import streamlit as st
from PIL import Image
from random import randint, random
from urllib.request import urlopen

import icmbio.icmbio_search as bio




app = st.sidebar.radio("Application:", ("ICMBio Visualizer", "Provador Oficial", "Rubia Photoshop"))



#########################################################################################################################
#
# PROVADOR OFICIAL
#
#########################################################################################################################

if app == "Rubia Photoshop":

    def load_image(url):
        img = Image.open(urlopen(url))
        img.load()
        array = np.asarray(img, dtype="uint8")
        return img, array

    def save_image(array, file):
        img = Image.fromarray(np.asarray(array, dtype="uint8"))
        img.save(file)

    def blur_image(layer, blur):
        blurred = np.copy(layer) # let's begin with a fresh copy of array
        # the max amount of blur is done by the minimum dimension
        for color in range(layer.shape[2]):
            for i in range(layer.shape[0]): 
                for j in range(layer.shape[1]):
                    sxm, sxM, sym, syM = i-blur, i+blur, j-blur, j+blur
                    if sxm < 0: sxm = 0
                    if sxM > layer.shape[0]: sxM = layer.shape[0]
                    if sym < 0: sym = 0
                    if syM > layer.shape[1]: syM = layer.shape[1]
                    sector = layer[sxm:sxM,sym:syM,color]
                    pixel_mean = int(sector.mean())
                    blurred[i,j,color] = pixel_mean
        return blurred
        
    def pixelate_image(layer, noise):
        dims = layer.shape
        noise_pixels = noise / 100 * 255 * 2 * (np.random.random((dims)) - 0.5)
        noise_pixels = noise_pixels.astype('int')
        noised = layer + noise_pixels
        out = np.where(noised > 255, 1, np.where(noised < 0, 1, 0))
        notout = np.where(out == 1, 0, 1)
        return out*layer + notout*noised

    def filter_image(layer, filter):
        filtered = (100 - filter) / 100 * layer
        filtered = filtered.astype('int')
        return filtered
    

    foto = st.sidebar.text_input("Image URL: ", "https://www.senaipa.org.br/themes/theme1/images/sitekit/index_.jpg")
    # get all fotos in the project folder
    #fotos = [os.path.join(file) for file in os.listdir() if file.endswith(".jpg")]
    #foto = st.sidebar.selectbox("Please choose a photo to adjust: ", fotos)

    # get changes to be made to the output image
    max_dimension = st.sidebar.slider("Please set image max dimension: ", 50, 500, 100)
    transparencia = st.sidebar.slider("Please set an image transparency: ", 0, 100, 20)
    blur = st.sidebar.slider("Please set a blur radius: ", 1, 20, 1)
    noise_r = st.sidebar.slider("Please set a noise level for red: ", 0, 100, 0)
    noise_g = st.sidebar.slider("Please set a noise level for green: ", 0, 100, 0)
    noise_b = st.sidebar.slider("Please set a noise level blue: ", 0, 100, 0)
    filter_r = st.sidebar.slider("Please set a filter level for red: ", 0, 100, 10)
    filter_g = st.sidebar.slider("Please set a filter level for green: ", 0, 100, 0)
    filter_b = st.sidebar.slider("Please set a filter level blue: ", 0, 100, 0)


    if foto:

        # read image and resize its dimensions according to max dimension setting
        img, array = load_image(foto)
        foto_largura, foto_altura = img.size
        if foto_largura > foto_altura:
            largura = max_dimension
            altura = int(foto_altura / foto_largura * largura)
        else:
            altura = max_dimension
            largura = int(foto_largura / foto_altura * altura)    
        img_resized = img.resize((largura, altura), Image.ANTIALIAS)
        novo_array = np.asarray(img_resized, dtype="uint8")
        

        # image manipulation
        
        # apply blur effect to the image
        novo_array = blur_image(novo_array, blur)
        
        # apply filter to the image
        channel_r = filter_image(novo_array[:,:,0], filter_r)
        channel_g = filter_image(novo_array[:,:,1], filter_g)
        channel_b = filter_image(novo_array[:,:,2], filter_b)

        # apply noise to the image
        channel_r = pixelate_image(channel_r, noise_r)
        channel_g = pixelate_image(channel_g, noise_g)
        channel_b = pixelate_image(channel_b, noise_b)

        # apply alpha layer
        channel_a = np.full((channel_r.shape[0],channel_r.shape[1]), int(255 * (100 - transparencia) / 100))
        novo_array = np.array([channel_r, channel_g, channel_b, channel_a])
        novo_array = np.stack(novo_array, axis=2)


        #show generated image
        foto_to_show = st.radio("Please select an image to show", ("Original", "Transformed"), 1)
        st.write(foto_to_show)
        if foto_to_show == "Original": st.image(img_resized)
        if foto_to_show == "Transformed": st.image(novo_array)

        # save new generated image to file into the folder generated/
        pasta = st.text_input("Please inform the complete folder path (ex. C:\myfolder\)", "C:\\")
        nova_foto = st.text_input("New file name", str(randint(10000,99999)))
        if st.button("SALVAR"):
            st.warning(pasta + nova_foto + ".png")
            save_image(novo_array, pasta + nova_foto + ".png")




#########################################################################################################################
#
# PROVADOR OFICIAL
#
#########################################################################################################################

if app == "Provador Oficial":

    experimentos_file = "provador/experimentos.csv"
    colunas = ["cur_ID", "data", "descricao", "responsavel", "produtos", "titulo", "status"]
    indice = "cur_ID"

    rotina = st.sidebar.radio("Escolha a rotina:", ("Gerenciar experimentos", "Formulário de análise", "Analisar resultados"))

    if rotina == "Gerenciar experimentos":
        st.title("Gerenciar experimentos")

        acao = st.radio("O que deseja fazer:", ("Criar experimento", "Consultar um experimento"))
        experimentos = pd.read_csv(experimentos_file, sep=";", header=0, index_col=indice, names=colunas)
        cur_ID = len(experimentos) + 1

        if acao != "Criar experimento":
            st.write("Selecione um experimento existente")
            status = st.selectbox("Status", ["Todos"] + experimentos["status"].unique().tolist())
            filtered = experimentos[experimentos["status"] == status] if status != "Todos" else experimentos
            responsavel = st.selectbox("Responsável", ["Todos"] + filtered["responsavel"].unique().tolist())
            filtered = filtered[filtered["responsavel"] == responsavel] if responsavel != "Todos" else filtered
            titulo = st.selectbox("Título", ["Todos"] + filtered["titulo"].unique().tolist())
            filtered = filtered[filtered["titulo"] == titulo] if titulo != "Todos" else filtered
            ID = st.selectbox("Código", ["Todos"] + filtered.index.tolist())
            if ID != "Todos": filtered = filtered.loc[ID]
            st.dataframe(filtered)
        else:
            data = st.date_input("Data:")
            responsavel = st.text_input("Responsável:")
            titulo = st.text_input("Título:")
            descricao = st.text_area("Descrição:")
            descricao = '\n'.join([descricao for descricao in descricao.split("\n")])
            produtos = st.text_area("Produtos: (um por linha)")
            produtos = '|'.join([produto for produto in produtos.split("\n")])
            status = "Aberto"
            novo_registro = pd.Series([data, descricao, responsavel, produtos, titulo, status], index=colunas[1:])
            novo_registro = pd.DataFrame(novo_registro).T
            experimentos = pd.concat([experimentos, novo_registro], ignore_index=True)
            st.write(experimentos)
            if data == "" or descricao == "" or responsavel == "" or produtos == "" or titulo == "":
                st.warning("Todos os campos são obrigatórios")
            else:
                if st.button("SALVAR"):
                    experimentos.to_csv(experimentos_file, sep=";", header=True)
        

    if rotina == "Formulário de análise":
        st.title("Formulário de análise")

    if rotina == "Analisar resultados":
        st.title("Analisar resultados")                




#########################################################################################################################
#
# ICMBIO
#
#########################################################################################################################

if app == "ICMBio Visualizer":

    st.title("ICMBio Visualizer")
    st.sidebar.header("Settings")

    key = '09aadb1b1d8840acacfa0fcece0acb13'
    key = st.sidebar.text_input("Product key", key)

    if st.sidebar.radio("Source of CSV database", ("Samples","Web")) == "Samples":
        FILES = [os.path.join(file) for file in os.listdir('icmbio/Arquivos_csv/') if file.endswith(".csv")]
        url = 'icmbio/Arquivos_csv/' + st.sidebar.selectbox("Please choose a database: ", np.sort(FILES))
    else:
        url = st.sidebar.text_input("URL for the CSV file")

    st.write("Reading file %s" % url)

    TAXONOMY_COLUMNS = ['Filo', 'Classe', 'Ordem', 'Familia', 'Genero', 'Especie']
    TAXONOMY_COLUMNS = st.sidebar.multiselect("Taxonomy columns to analyse", TAXONOMY_COLUMNS, TAXONOMY_COLUMNS)

    LOCATION_COLUMNS = ['Pais', 'Estado/Provincia', 'Municipio', 'Localidade', 'Latitude', 'Longitude']
    LOCATION_COLUMNS = st.sidebar.multiselect("Location columns to analyse", LOCATION_COLUMNS, LOCATION_COLUMNS)

    # class initializer
    biodiversity = bio.getBiodiversity(url, key, TAXONOMY_COLUMNS, LOCATION_COLUMNS)

    # missing data analysis
    biodiversity.checkEmpty()
    if st.checkbox("Show missing data statistics (% of data missing)"):
        st.dataframe(biodiversity.df_dataNAN)

    # run taxonomic analysis
    biodiversity.getTaxonomy(col_name='Nível Taxonômico')
    if st.checkbox("Show taxonomic data (%d rows x %d columns)" % (biodiversity.df_taxonomy.shape[0],biodiversity.df_taxonomy.shape[1])):
        st.dataframe(biodiversity.df_taxonomy)

    # filtering data to show
    FILTER_FIELDS = st.sidebar.multiselect("Please select one or more columns to filter by", list(biodiversity.df_data.columns))
    FILTER_VALUES = [st.sidebar.multiselect("Filter values for column %s"%column, np.sort(biodiversity.df_data[column].unique())) for column in FILTER_FIELDS]
    biodiversity.filterFields(FILTER_FIELDS, FILTER_VALUES)
    biodiversity.getTaxonomy(col_name='Nível Taxonômico')
    if st.checkbox("Show filtered data (%d rows x %d columns)" % (biodiversity.df_filtered.shape[0],biodiversity.df_filtered.shape[1])):
        st.dataframe(biodiversity.df_filtered)

    # check if latitude and longitude are correct or not
    points_to_plot = int(0.1*biodiversity.df_filtered.shape[0]+1) if int(0.1*biodiversity.df_filtered.shape[0]+1) < 20 else 20
    LOCATION_SAMPLING = st.sidebar.slider("Number of samples to plot", 1, biodiversity.df_filtered.shape[0], points_to_plot)
    biodiversity.checkCoordinates(LOCATION_SAMPLING)
    if st.checkbox("Show locations sample data (%d rows x %d columns)" % (biodiversity.df_location_sample.shape[0],biodiversity.df_location_sample.shape[1])):
        if len(biodiversity.STOP_WORDS) < 2: st.write("Please check if your stopwords.txt is in the project folder")
        st.write(biodiversity.df_location_sample[["lat", "lon", "ReportedAddress", "ReversedAddress", "Similarity"]])

    # show map with sampled observations
    dfmap = biodiversity.df_location_sample[["lat","lon","ReportedAddress","ReversedAddress","Similarity"]].copy()
    dfmap["colorR"] = dfmap["Similarity"].apply(lambda x: float((100 - x) / 100 * 255) + 0.01)
    dfmap["colorG"] = dfmap["Similarity"].apply(lambda x: float(x / 100 * 255) + 0.01)
    dfmap["colorB"] = dfmap["Similarity"].apply(lambda x: 0.01)
    dfmap["radius"] = dfmap["Similarity"].apply(lambda x: 1000 * x + 0.01)

    try:
        rangelat = math.log2(170 / (dfmap['lat'].max()-dfmap['lat'].min()))
        rangelon = math.log2(360 / (dfmap['lon'].max()-dfmap['lon'].min()))
        zoom = int(min(rangelat, rangelon)) + 1
    except:
        zoom = 10

    st.deck_gl_chart(
        viewport={
            'latitude': dfmap['lat'].median(),
            'longitude': dfmap['lon'].median(),
            'zoom': zoom,
            'pitch': 50,
            'opacity': 0.1
        },
        layers = [{
            'type': 'ScatterplotLayer',
            'data': dfmap,
            'opacity': 0.5,
            'pickable': True,
            'autoHighlight': True,
            'stroked': True,
            #'getRadius': 5000,
            #'filled': True,
            'radiusScale': 1,
            'radiusMinPixels': 3, 
            'radiusMaxPixels': 30,
            'getLineColor': [220, 220, 220],
            'lineWidthMinPixels': 1,
            #'onHover': 'find documentation on how to implement this
    }])