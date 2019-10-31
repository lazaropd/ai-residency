
import pandas as pd
import numpy as np

import textdistance
import folium
import string
import os
import re

from statistics import mean 
from opencage.geocoder import OpenCageGeocode
from unidecode import unidecode

# remove special characters and stopwords from addresses
STR_SPECIALCHAR = "[({;:,<>_+=/.\"?!\t\r})]"

class getBiodiversity():

    def __init__(self, url, key, taxonomy_columns, location_columns):
        self.url = url
        self.geocoder = OpenCageGeocode(key)
        self.TAXONOMY_COLUMNS = taxonomy_columns
        self.LOCATION_COORDINATES = location_columns
        try:
            self.df_data = pd.read_csv(url, sep=';', header=0, encoding='utf-8')
        except Exception as e:
            self.df_data = pd.DataFrame()
            print("Aborting... couldn't read this file: %s" % url)
            print (e.args)
        return None
    
    def getColumns(self):
        self.df_columns = list(self.df_data.columns)
        return None
    
    def checkEmpty(self):
        self.getColumns()
        self.df_dataNAN = pd.DataFrame(np.where((self.df_data == '') | (self.df_data == 'Sem Informações'), 1, 0))
        self.df_dataNAN.columns = self.df_columns
        self.df_dataNAN = 100*self.df_dataNAN.mean()
        return None

    def getLastFilled(self, columns):
        filled_columns = [column for column in columns if (column != "Sem Informações")]
        return 'NA' if len(filled_columns) == 0 else self.TAXONOMY_COLUMNS[len(filled_columns)-1]
    
    def addTaxonomicLevel(self, col_name):
        self.df_data[col_name] = self.df_data[self.TAXONOMY_COLUMNS].apply(lambda x: self.getLastFilled(x), axis=1)
        self.df_taxonomy_info =  self.df_data[col_name].value_counts()
        return None

    def extractTaxonomy(self, columns):
        self.df_taxonomy = self.df_data[columns].copy()
        return None
    
    def getTaxonomy(self, col_name='taxonomic_level'):
        self.addTaxonomicLevel(col_name)
        self.extractTaxonomy(self.TAXONOMY_COLUMNS+[col_name])
        return None
    
    def filterFields(self, columns, values):
        #filter = np.logical_and.reduce([self.df_data[columns[i]].isin(values[i]) for i in range(len(columns))])
        filter = np.sum([self.df_data[columns[i]].isin(values[i])+(len(values[i])==0) for i in range(len(columns))], axis=0) == len(columns)
        if columns: 
            self.df_filtered = self.df_data[filter].copy()
        else:
            self.df_filtered = self.df_data.copy()
        return None
    
    def parseFloat(self, info):
        value = float(info)
        try:
            value = float(info)
        except:
            value = 0.0
        return value
    
    """
    def checkGeoInfo(self, components, reported):
        aux = []
        unmatched = 0
        for elem in ["country", "state", "state_code", "city"]:
            try:
                value = components[elem]
            except:
                value = "NA"
            aux.append(value)
        unmatched += 1 if reported[0] != aux[0] else 0
        unmatched += 1 if not reported[1] in [aux[1], aux[2]] else 0
        unmatched += 1 if reported[2] != aux[3] else 0
        return unmatched
    """
    
    def removeStopWords(self, address):
        address = re.sub(STR_SPECIALCHAR, ' ', address).lower()
        address = ' '.join([word.strip(' ') for word in address.split(' ') if word.strip(' ') not in self.STOP_WORDS])
        return address

    def removeNonAscii(self, text):
        return unidecode(str(text))

    def reverseGeocode(self, latlon):
        geo = self.geocoder.reverse_geocode(latlon[0], latlon[1], no_annotations = '1', pretty = '1', language='pt')
        #comp = geo[0]['components']
        #similarity = self.checkGeoInfo(comp, [latlon[2], latlon[3], latlon[4]])
        reversed = self.removeStopWords(geo[0]['formatted'])
        reported = self.removeStopWords(' '.join(latlon[2:6]))
        reversed = self.removeNonAscii(reversed)
        reported = self.removeNonAscii(reported)
        similarity = 100 * textdistance.jaccard(reported , reversed) 
        return pd.Series((reported, reversed, similarity))
    
    def setMapZoom(self, coords):
        try:
            rangelat = math.sqrt(170 / (max(coords[0][:])-min(coords[0][:])))
            rangelon = math.sqrt(360 / (max(coords[1][:])-min(coords[1][:])))
            zoom = int(min(rangelat, rangelon)) + 1
        except:
            zoom = 1
        return zoom
    
    """
    def printMap(self):
        coords = self.df_location_sample[["lat", "lon", "ReversedAddress", "Similarity"]].T.values.tolist()
        COLORS = ['green', 'lightgreen', 'orange', 'red']
        center = [mean(coords[0][:]), mean(coords[1][:])]
        zoom = self.setMapZoom(coords[0:2][:])
        my_map = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")
        for i in range(len(self.df_location_sample)):
            folium.Marker(location=[coords[0][i], coords[1][i]], popup=coords[2][i], 
                          icon=folium.Icon(color=COLORS[coords[3][i]], icon='map-marker')).add_to(my_map) 
        self.observations_map = my_map
        return None
    """
        
    def checkCoordinates(self, size):
        try:
            stopwords = open("stopwords.txt")
            self.STOP_WORDS = [linha.rstrip(" ").rstrip("\n") for linha in stopwords.readlines()]
        except:
            self.STOP_WORDS = ["asdfasdfasdf"] # if stopwords file not found
        self.STOP_WORDS = [sw.lower().strip() for sw in set(self.STOP_WORDS)]
        self.df_filtered["lat"] = self.df_data["Latitude"].apply(lambda x: self.parseFloat(x))
        self.df_filtered["lon"] = self.df_data["Longitude"].apply(lambda x: self.parseFloat(x))
        if len(self.df_filtered) < size:
            print("Not enough data to show. Please check your filter opetions")
            self.df_location_sample = pd.DataFrame()
            self.observations_map = None
            return None
        self.df_location_sample = self.df_filtered.sample(n=size).copy()
        self.df_location_sample[['ReportedAddress','ReversedAddress','Similarity']] = self.df_location_sample[['lat','lon']+self.LOCATION_COORDINATES].apply(self.reverseGeocode, axis=1)
        #self.printMap()
        return None
