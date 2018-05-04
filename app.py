from flask import Flask, render_template, request
import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import Reader, Dataset, SVD


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/send', methods=['POST','GET'])
def send():
    if request.method == 'POST':
     movie = request.form['movie']
     movies = find(movie)
     return render_template('final.html', movies = movies)


def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
    

def find(movie):
   mov_data = pd. read_csv('movies.csv')
#print(mov_data.head(1))
   mov_data= mov_data.drop_duplicates(subset='title')
   mov_data['genres'] = mov_data['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

#print(mov_data['genres'])

   mov_data['year'] = pd.to_datetime(mov_data['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

   s = mov_data.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
   s.name = 'genre'
   gen_mov_data = mov_data.drop('genres', axis=1).join(s)
#print(gen_mov_data.head(1))
   title_movie = gen_mov_data[gen_mov_data['title'] == movie]
   
   #Title not present in dataset
   if not len(title_movie):
       unavail = ['Movie title Unavailable. Try another']
       return unavail
    #print(title_movie['genre'])
   genre_titles = gen_mov_data[gen_mov_data['genre'].isin(title_movie['genre'])]
    #print(genre_titles.head(2))
    #print(genre_titles.shape)
    
   vote_counts = genre_titles[genre_titles['vote_count'].notnull()]['vote_count'].astype('int')
   vote_averages = genre_titles[genre_titles['vote_average'].notnull()]['vote_average'].astype('int')
   C = vote_averages.mean()
   m = vote_counts.quantile(0.90)
    
   qualified = genre_titles[(genre_titles['vote_count'] >= m) & (genre_titles['vote_count'].notnull()) & (genre_titles['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
   qualified['vote_count'] = qualified['vote_count'].astype('int')
   qualified['vote_average'] = qualified['vote_average'].astype('int')
    
   qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
   qualified = qualified.sort_values('wr', ascending=False)
    
   rec_titles = qualified[qualified['title'] == movie]
   great_five = qualified[qualified['wr'] > rec_titles['wr'].iloc[0]].sort_values('wr', ascending=True)
   less_five = qualified[qualified['wr'] < rec_titles['wr'].iloc[0]].sort_values('wr', ascending=False)
   
   print("\n \n Here is your Basic recommendation!")
   p = ['Basic Recommendation']
    
   t = 'Noname'
   i = 0
   for x in range(0, 5):
        while t == great_five['title'].iloc[i]:
            i= i+1
        print(great_five['title'].iloc[i],"-->", great_five['wr'].iloc[i])
        
        i = i+1
        t = great_five['title'].iloc[i]
        p.append(t)
        
        
   # print(rec_titles['title'].iloc[0],"-->", rec_titles['wr'].iloc[0])
   t = 'Noname'
   i = 0
   for x in range(0, 5):
        while t == less_five['title'].iloc[i]:
            i= i+1
        print(less_five['title'].iloc[i], "-->" , less_five['wr'].iloc[i])
        #p = "{0}  nl {1}      {2}".format(p,less_five['title'].iloc[i], less_five['wr'].iloc[i])
        
        i = i+1
        t = less_five['title'].iloc[i]
        p.append(t)
    
   
   p.append('Content Recommendation')
   print("\n \nContent Recommendation")
   
   links_small = pd.read_csv('links.csv')
   links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

   #mov_data = mov_data.drop([19730, 29503, 35587])
   mov_data['id'] = mov_data['id'].astype('int')
   smov_data = mov_data[mov_data['id'].isin(links_small)]
#print(smov_data.shape)
   
   drctr = pd. read_csv('credits.csv')
   drctr['crew'] = drctr['crew'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x if i['job'] == 'Director'])
   title_id = mov_data[mov_data['title'] == movie]['id']
   find_director = drctr[drctr['id'].isin(title_id)]['crew']
   st = str(find_director)
   director = "".join([i for i in st if not i.isdigit()])
   
   kywds = pd. read_csv('keywords.csv')
   kywds['keywords'] = kywds['keywords'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
   find_keyword = kywds[kywds['id'].isin(title_id)]['keywords']
   ky = str(find_keyword)
   keywords = "".join([i for i in ky if not i.isdigit()])
   
   smov_data['tagline'] = smov_data['tagline'].fillna('')
   smov_data['title'] = smov_data['title'].fillna('')
   smov_data['description'] = smov_data['overview'] + smov_data['tagline'] + smov_data['title'] + keywords + director
#   for i in range(1, 100):
#       smov_data['description'] = smov_data['description'] + director
   smov_data['description'] = smov_data['description'].fillna('')
   #print(smov_data['description'].head())

   tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
   tf_mat = tf.fit_transform(smov_data['description'])

   #print(tf_mat.head(2))

   cosine_sim = linear_kernel(tf_mat, tf_mat)

#print(cosine_sim[0])

   smov_data = smov_data.reset_index()
   titles = smov_data['title']
   indices = pd.Series(smov_data.index, index=smov_data['title'])
   
   idx = indices[movie]
    #print(idx)
   sim_scores = list(enumerate(cosine_sim[idx]))
    #print(sim_scores)
   sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   sim_scores = sim_scores[1:11]
   movie_indices = [i[0] for i in sim_scores] 
   scores = [i[1] for i in sim_scores] 
   #p.append(titles.iloc[movie_indices])
   
   for x in titles.iloc[movie_indices]:
       print(x)
       p.append(x)

   print("\nCosine Scores")
   for x in scores:
       print(x)
       
   p.append('Hybrid Recommendation')
   
   print("\n \nHybrid Recommendation\n \n")
   smov_data = smov_data[smov_data['title'].isin(qualified['title'])]
   #print(smov_data.head(5))
   tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
   tf_mat = tf.fit_transform(smov_data['description'])

#print(tfidf_matrix.shape)

   cosine_sim = linear_kernel(tf_mat, tf_mat)

#print(cosine_sim[0])

   smov_data = smov_data.reset_index()
   titles = smov_data['title']
   indices = pd.Series(smov_data.index, index=smov_data['title'])
   
   idx = indices[movie]
    #print(idx)
   sim_scores = list(enumerate(cosine_sim[idx]))
    #print(sim_scores)
   sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   
   
   ratings = pd.read_csv('sratings.csv')
#print(ratings.head())

   reader = Reader()
   data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
#data.split(n_folds=5)
   algo = SVD()
#evaluate(algo, data, measures=['RMSE', 'MAE'])

   trainset = data.build_full_trainset()
   algo.train(trainset)
   
   hyb = pd.read_csv('links.csv')[['movieId', 'tmdbId']]
   hyb['tmdbId'] = hyb['tmdbId'].apply(convert_int)
   hyb.columns = ['movieId', 'id']
   hyb = hyb.merge(smov_data[['title', 'id']], on='id').set_index('title')

   index_map = hyb.set_index('id')
    
   idx = indices[movie]
    
   sim_mat = list(enumerate(cosine_sim[int(idx)]))
   sim_mat = sorted(sim_mat, key=lambda x: x[1], reverse=True)
   sim_mat = sim_mat[1:15]
   movie_indices = [i[0] for i in sim_mat]
    
   #Predict the movie user 500 will like using Collaberative Filtering
   
   movies = smov_data.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'id']]
   movies['est'] = movies['id'].apply(lambda x: algo.predict(500, index_map.loc[x]['movieId']).est)  #UserId is 500
   movies = movies.sort_values('est', ascending=False)
   print(movies['title'].head(10))  
   print(movies['est'].head(10))
   
   print(title_movie['genre'])
   
   for x in movies['title'].head(10):
       p.append(x)
       print(x)
    
   return p



if __name__ == '__main__':
    app.run()
