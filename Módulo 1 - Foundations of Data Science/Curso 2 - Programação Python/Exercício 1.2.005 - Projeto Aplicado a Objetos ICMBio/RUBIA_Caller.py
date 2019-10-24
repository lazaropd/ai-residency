
from importlib import reload
import RUBIA; reload(RUBIA)


LOCATION_SAMPLING = 10 # number of locations to check and plot in the map

url = "portalbio_export_16-10-2019-14-39-54.csv"


def special_print(title, content):
    #display(Markdown("### %s:" % title)) # uncomment this when using notebook
    print(title + ": ") # uncomment this when using VSCode or other IDE
    print(content)
    return None


####################################################################################
#
# Class manipulation and callers


# create a new instance of the class
biodiversity = RUBIA.getBiodiversity(url)

# check for and report statistics about missing data
biodiversity.checkEmpty()

# check for taxonomic available info and add it to the dataframe
biodiversity.getTaxonomy(col_name='Nível Taxonômico')

# apply filters and return a filtered dataframe
filter_columns = ['Municipio','Filo']
filter_values = [['Vitoria','Niquelândia','Nova Friburgo','Vitoria','Natal'],
                ['Mollusca','Annelida','Magnoliophyta']]
biodiversity.filterFields(filter_columns, filter_values)

# apply reverse geocode function and plot map of sampled observations
biodiversity.checkCoordinates(LOCATION_SAMPLING)



####################################################################################
#
# Show sample of each output - raw data load

special_print("File URL", biodiversity.url)
special_print("Raw file info", biodiversity.data_info)
special_print("Raw file sample", biodiversity.df_data.head(1).T)
special_print("Dataframe columns", biodiversity.df_columns)


####################################################################################
#
# Show sample of each output - data missing analysis

special_print("Data missing sample (1 = missing)", biodiversity.df_dataNAN.head(5).T)
special_print("Data missing statistics (%)", biodiversity.df_data_missing)


####################################################################################
#
# Show sample of each output - show taxonomic info

special_print("Raw data sample after taxonomic level inclusion", biodiversity.df_data.head(1).T)
special_print("Taxonomic info", biodiversity.df_taxonomy_info)
special_print("Taxonomy sample", biodiversity.df_taxonomy.head(3).T)


####################################################################################
#
# Show sample of each output - filtered data

special_print("Filtered data info", biodiversity.filtered_info)
special_print("Filtered data sample", biodiversity.df_filtered.head(1).T)


####################################################################################
#
# Show sample of each output - show location info

special_print("Sample of locations to check", biodiversity.df_location_sample.head(1).T)


####################################################################################
#
# Show sample of each output - show map with reported observations

print("Observations (click to see more details)")
biodiversity.observations_map



