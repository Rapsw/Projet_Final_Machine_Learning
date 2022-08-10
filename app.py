from unittest import result
import streamlit as st 
import pandas as pd 
#import numpy as np
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
#import plotly.express as px 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#import seaborn as sns
import pickle
from PIL import Image  



st.title("Acheter l'appartement de vos rêves au meilleur prix du marché") 

# Image
img = Image.open("maison.jpg")
st.image(img, width=700)  


# surface
sqft_living = st.number_input('Surface', key ='sqft_living')
st.write('La surface est:', sqft_living)


 # bedrooms
bedrooms = st.number_input('Nombre de chambres', value=1, key = 'bedrooms')
st.write('chambres:', bedrooms)


# bathrooms
bathrooms = st.selectbox(
     'Salle de bain ',
     ('0', '1', '2','3','4','5','6', '7'),  index = 1, key ='bathrooms')
st.write('Salle de bain:', bathrooms)


# sqft_lot
sqft_lot = st.slider('Terrain', 0, 1000000, 10000,  key ='sqft_lot')
st.write('Le terrain est:', sqft_lot)


# floors
floors = st.selectbox(
     "Nombre d'étages ",
     ('0', '1', '2','3'), index = 1, key ='floors')
st.write('Etages:', floors)


# waterfront
waterfront = st.radio(
     "Vue sur la mer ",
     ('Oui', 'Non') , key = 'waterfront')
if waterfront == 'Yes': 
    waterfront = 1
else: 
    waterfront = 0


# view
view = st.selectbox(
     "La maison a été visitée ",
     ('0', '1', '2','3','4'), index = 1,  key = 'view')
st.write('Visitée:', view)


# condition	
condition = st.selectbox(
     "L'état de la maison ",
     ('0', '1', '2','3','4', '5'), index = 2  , key = 'condition')
st.write("L'état:", condition)


# grade
grade = st.selectbox(
     "Note de qualité et design ",
     ('0', '1', '2','3','4','5','6','7','8','9','10','11','12','13'), index = 6,  key = 'grade')
st.write('Note:', grade)


# sqft_above
sqft_above = st.slider('Surface sans cave', 0, 10000, 2500,  key = 'sqft_above' )
st.write('La surface est:', sqft_above)


# sqft_basement
sqft_basement = st.slider('Cave', 0, 5000, 500,  key = 'sqft_basement')
st.write('Cave :', sqft_basement)


# yr_built
yr_built = st.slider('Année de construction ', 1900, 2030, 1992,  key = 'yr_built')
st.write('Année:', yr_built)


# yr_renovated
yr_renovated = st.slider('Année de rénovation ', 1900, 2030, 0,  key = 'yr_renovated')
st.write('Année:', yr_renovated)


# zipcode 
zipcode = st.selectbox(
     'Code postal',
     ("79236",
"98001",
"98002",
"98003",
"98004",
"98005",
"98006",
"98007",
"98008",
"98009",
"98010",
"98011",
"98013",
"98014",
"98015",
"98019",
"98022",
"98023",
"98024",
"98025",
"98027",
"98028",
"98029",
"98030",
"98031",
"98032",
"98033",
"98034",
"98035",
"98038",
"98039",
"98040",
"98041",
"98042",
"98045",
"98047",
"98050",
"98051",
"98052",
"98053",
"98055",
"98056",
"98057",
"98058",
"98059",
"98062",
"98063",
"98064",
"98065",
"98070",
"98071",
"98072",
"98073",
"98074",
"98075",
"98077",
"98083",
"98089",
"98092",
"98093",
"98101",
"98102",
"98103",
"98104",
"98105",
"98106",
"98107",
"98108",
"98109",
"98111",
"98112",
"98113",
"98114",
"98115",
"98116",
"98117",
"98118",
"98119",
"98121",
"98122",
"98124",
"98125",
"98126",
"98127",
"98129",
"98131",
"98133",
"98134",
"98136",
"98138",
"98139",
"98141",
"98144",
"98145",
"98146",
"98148",
"98154",
"98155",
"98158",
"98160",
"98161",
"98164",
"98165",
"98166",
"98168",
"98170",
"98174",
"98175",
"98177",
"98178",
"98181",
"98185",
"98188",
"98190",
"98191",
"98194",
"98195",
"98198",
"98199",
"98224",
"98288"),  key = 'zipcode')
st.write('Le code postal est:', zipcode)

# lat 
lat = st.number_input('Latitude', value = 47, key = 'lat')
st.write('La surface est:', lat)


# sqft_living15
sqft_living15 = st.slider('Surface voisins', 0, 20000,10000,  key = 'sqft_living15')
st.write('La surface est:', sqft_living15)


# sqft_lot15
sqft_lot15 = st.slider('Terrain voisins', 0, 300000, 100000, key = 'sqft_lot15')
st.write('Le terrain est:', sqft_lot15)




data = {
    'sqft_living':sqft_living,
    'bedrooms': bedrooms,
    'bathrooms':bathrooms,
    'sqft_lot':sqft_lot,
    'floors':floors,
    'waterfront':waterfront,
    'view':view,
    'condition':condition,
    'grade':grade,
    'sqft_above':sqft_above,
    'sqft_basement':sqft_basement,
    'yr_built':yr_built,
    'yr_renovated':yr_renovated,
    'zipcode':zipcode,
    'lat':lat,
    'sqft_living15':sqft_living15,
    'sqft_lot15': sqft_lot15

}

parametres = pd.DataFrame(data, index=[0])

pickle_in = open('my_pipe_lasso.pkl', 'rb') 
my_pipe_lasso = pickle.load(pickle_in)

print(parametres.info())


def prediction(parametres):

    prediction = my_pipe_lasso.predict(parametres)
    print(prediction)
    return prediction

if st.button('Prédiction'):
    result = prediction(parametres)
    prix = round(result[0],1)
    st.write('## Prix de votre maison est :',prix, "$")

