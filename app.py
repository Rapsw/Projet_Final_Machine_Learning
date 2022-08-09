from unittest import result
import streamlit as st 
import pandas as pd 
import numpy as np
import pickle 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
import plotly.express as px 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import seaborn as sns
import pickle
from PIL import Image  



st.title("Acheter l'appartement de vos rêves au meilleur prix du marché") 

# Image

img = Image.open("maison.jpg")
st.image(img, width=700)  

# surface

sqft_living = st.number_input('Surface')
st.write('La surface est', sqft_living)

 # bedrooms

bedrooms = st.selectbox(
     'Nombre de chambres ',
     ('0', '1', '2','3','4','5','6', '7', '8','9','10', '11','12'))
st.write('Chambres:', bedrooms)

# bathrooms

bathrooms = st.selectbox(
     'Nombre de chambres ',
     ('0', '1', '2','3','4','5','6', '7'))
st.write('Salle de bain:', bathrooms)

# sqft_lot

sqft_lot = st.number_input('Terrain')
st.write('Le terrain est', sqft_lot)

# floors

floors = st.selectbox(
     "Nombre d'étages ",
     ('0', '1', '2','3'))
st.write('Etages:', floors)

# waterfront

waterfront = st.radio(
     "Vue sur la mer ",
     ('Oui', 'Non'))

# view

view = st.selectbox(
     "La maison a été visitée ",
     ('0', '1', '2','3','4'))
st.write('Visitée:', view)

# condition	

condition = st.selectbox(
     "L'état de la maison ",
     ('0', '1', '2','3','4', '5'))
st.write("L'état:", condition)

# grade

grade = st.selectbox(
     "Note de qualité et design ",
     ('0', '1', '2','3','4','5','6','7','8','9','10','11','12','13'))
st.write('Note:', grade)

# sqft_above

sqft_above = st.number_input('Surface sans cave')
st.write('La surface est:', sqft_above)

# sqft_basement

sqft_basement = st.number_input('Cave')
st.write('Cave :', sqft_basement)

# yr_built

yr_built = st.slider('Année de construction ', 1900, 2030, 2000)
st.write('Année:', yr_built)

# yr_renovated

yr_renovated = st.slider('Année de rénovation ', 1900, 2030, 2022)
st.write('Année:', yr_renovated)

# zipcode 

zipcode = st.selectbox(
     'Code postal ',
     ())
st.write('Code:', zipcode)

# lat 

lat = st.number_input('Latitude')
st.write('La surface est:', lat)

# sqft_living15

sqft_living15 = st.number_input('Surface voisins')
st.write('La surface est:', sqft_living15)

# sqft_lot15

sqft_lot15 = st.number_input('Terrain voisins')
st.write('Le terrain est:', sqft_lot15)

#year

# predict 


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


def prediction(parametres):

    prediction = my_pipe_lasso.predict(parametres)
    print(prediction)
    return prediction

if st.button('Prédiction'):
    result = prediction(parametres)
    st.write('Prix de votre maison :',result)

