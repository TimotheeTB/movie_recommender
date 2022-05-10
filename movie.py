import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import re
import os
# print(os.listdir("../input"))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
from PIL import Image
from linkpreview import link_preview


st.title('Movie Recommender')

# récupérer l'année dans le titre
def getYear(title):
    result = re.search(r'\(\d{4}\)', title)
    if result:
        found = result.group(0).strip('(').strip(')')
    else:
        found = 0
    return int(found)

# retirer l'année du titre
def removeyear(string):
    result=re.search(r'(\d{4})',string)
    if result:
        return string[:-6].strip()
    return string

# fusionner les features dans une même column pour chaque row
def get_important_features(data):
  important_features = []
  for i in range(0, data.shape[0]):
    important_features.append(data['title'][i]+' '+data['genres'][i])

  return important_features



st.header('Recommandations par film')

@st.cache(show_spinner=False, allow_output_mutation=True)
def imports():
    path =  r"C:/Users/Timothee TOUMANI/Desktop/work/work_projects/poc_dh/movie_reco"
    movies = pd.read_csv('{}/ml-latest-small/movies.csv'.format(path))
    movies["title2"]=movies.title.apply(lambda x:x.lower())
    ratings = pd.read_csv('{}/ml-latest-small/ratings.csv'.format(path))
    corrMatrix = pd.read_csv('{}/matrice2.csv'.format(path))
    corrMatrix.index = corrMatrix.movieId
    corrMatrix.drop(columns = 'movieId', inplace = True)
    corrMatrix.columns = corrMatrix.columns.astype(int)
    movies['year'] = movies.apply(lambda x: getYear(x['title']), axis=1)
    movies['title'] = movies.title.apply(removeyear)
    features = ['title','genres']
    movies['features'] = get_important_features(movies)
    cm = CountVectorizer().fit_transform(movies['features'])
    links = pd.read_csv('{}/ml-latest-small/links.csv'.format(path))
    return (movies,ratings,corrMatrix, cm, links)

movies,ratings,corrMatrix,cm, links=imports()

@st.cache(show_spinner=False)
def get_cosine_sim(cm):
    cs = cosine_similarity(cm)
    return cs

cs=get_cosine_sim(cm)


# content base

def userinput():
    title=st.text_input("Entrez le titre")
    return title.lower()

entry1=userinput()

@st.cache(show_spinner=False)
def pos(entry):
    possibilities=movies[movies.title2.str.contains(entry)].reset_index(drop=True)
    return possibilities

possibilities=pos(entry1)
if len(possibilities)==0:
    st.write("Aucun film ne correspond à la recherche")
else:
    if len(possibilities)==1:
        target=possibilities.title[0]
    else:
        choice=st.selectbox(label="Voulez-vous dire ?",options=possibilities.title)

        if st.button('Valider'):
            target=choice

    try :
        target_index=movies.loc[movies.title==target].index.values[0]

        # crée une liste avec [(movie id, similarity score), (...)]
        scores = list(enumerate(cs[target_index]))

        # trie la liste
        # x est le score
        # element en pos 1 est le score de similarité
        # reverse pour descending order
        # on prend tout sauf le premier car lui meme
        sorted_scores = sorted(scores, key = lambda x: x[1], reverse = True)
        sorted_scores = sorted_scores[1:]
        sorted_scores = np.array(sorted_scores)

        df = movies.loc[sorted_scores[:10,0],['title','movieId']]

        # loop pour afficher les x premiers
        j = 0

        for item in sorted_scores:
          movie_title = movies[movies.index == item[0]]['title'].values[0]
          st.write(j+1, movie_title)
          j = j+1
          if j >9:
            break
        image=links.loc[links.movieId.isin(df.movieId),"final_link"]
        liste=[]
        for i in image:
            pic=Image.open(urllib.request.urlopen(link_preview(i).absolute_image))
            liste.append(pic)

        st.image(liste,width=120)

    except :
        pass


st.header('Recommandations personalisées')
# collaborative filtering

def get_similar(movie_name,rating):
    similar_ratings = corrMatrix[movie_name]*(rating-2.5)
    similar_ratings = similar_ratings.sort_values(ascending=False)
    #print(type(similar_ratings))
    return similar_ratings



st.sidebar.header("Notes de l'utilisateur")

try :
    user = [("Amazing Spider-Man, The",int(st.sidebar.slider("The Amazing Spider-Man",0, 5))),
    ("Mission: Impossible III",int(st.sidebar.slider("Mission Impossible 3",0, 5))),
    ("Toy Story 3",int(st.sidebar.slider("Toy Story 3",0, 5))),
    ("2 Fast 2 Furious (Fast and the Furious 2, The)",int(st.sidebar.slider("2 fast 2 furious",0, 5))),
    ("Snatch",int(st.sidebar.slider("Snatch",0, 5))),
    ("Pulp Fiction",int(st.sidebar.slider("Pulp Fiction",0, 5))),
    ("Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark)",int(st.sidebar.slider("Indiana Jones 1",0, 5)))]
    similar_movies = pd.DataFrame()

    if st.sidebar.button('Confirmer'):

        for movie,rating in user:
            similar_movies = similar_movies.append(get_similar(movies.loc[movies.title==movie,"movieId"].values[0],rating),ignore_index = True)

        df2 = pd.DataFrame(movies.set_index('movieId', drop = True).loc[similar_movies.sum().sort_values(ascending=False).head(10).index,'title'])
        df2['movieId'] = df2.index

        st.write(df2['title'])
        image2=links.loc[links.movieId.isin(df2.movieId),"final_link"]
        liste2=[]
        for i in image2:
            pic2=Image.open(urllib.request.urlopen(link_preview(i).absolute_image))
            liste2.append(pic2)

        st.image(liste2,width=120)
except :
    pass