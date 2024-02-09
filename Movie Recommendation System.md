```python
import pandas as pd

# https://files.grouplens.org/datasets/movielens/ml-25m.zip
movies = pd.read_csv("movies.csv")
```


```python
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
    </tr>
  </tbody>
</table>
</div>




```python
import re

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title
```


```python
movies["clean_title"] = movies["title"].apply(clean_title)
```


```python
movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>clean_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
      <td>Adventure|Animation|Children|Comedy|Fantasy</td>
      <td>Toy Story 1995</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
      <td>Adventure|Children|Fantasy</td>
      <td>Jumanji 1995</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
      <td>Comedy|Romance</td>
      <td>Grumpier Old Men 1995</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
      <td>Comedy|Drama|Romance</td>
      <td>Waiting to Exhale 1995</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
      <td>Comedy</td>
      <td>Father of the Bride Part II 1995</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2))

tfidf = vectorizer.fit_transform(movies["clean_title"])
```


```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    
    return results
```


```python
import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data["new"]
        if len(title) > 1:
            display(search(title))

movie_input.observe(on_type, names='value')


display(movie_input, movie_list)
```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python
ratings = pd.read_csv("ratings.csv")
```


```python
ratings.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>296</td>
      <td>5.0</td>
      <td>1147880044</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>306</td>
      <td>3.5</td>
      <td>1147868817</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>307</td>
      <td>5.0</td>
      <td>1147868828</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>665</td>
      <td>5.0</td>
      <td>1147878820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>899</td>
      <td>3.5</td>
      <td>1147868510</td>
    </tr>
  </tbody>
</table>
</div>




```python
ratings.dtypes
```




    userId         int64
    movieId        int64
    rating       float64
    timestamp      int64
    dtype: object




```python
movie_id = 4896
```


```python
similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
similar_users
```




    array([     2,     20,    117, ..., 162508, 162524, 162538], dtype=int64)




```python
similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
similar_user_recs
```




    72             110
    74             151
    76             260
    79             318
    80             333
                 ...  
    24999756     81845
    24999761     93510
    24999762     93988
    24999769    102993
    24999776    116797
    Name: movieId, Length: 675001, dtype: int64




```python
similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

similar_user_recs = similar_user_recs[similar_user_recs > .10]
```


```python
all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
```


```python
all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
```


```python
rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
rec_percentages.columns = ["similar", "all"]
```


```python
rec_percentages
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similar</th>
      <th>all</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.276447</td>
      <td>0.124484</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.102564</td>
      <td>0.100096</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.187614</td>
      <td>0.144186</td>
    </tr>
    <tr>
      <th>50</th>
      <td>0.189856</td>
      <td>0.200119</td>
    </tr>
    <tr>
      <th>110</th>
      <td>0.192378</td>
      <td>0.160556</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>119145</th>
      <td>0.101723</td>
      <td>0.020931</td>
    </tr>
    <tr>
      <th>122886</th>
      <td>0.119938</td>
      <td>0.026926</td>
    </tr>
    <tr>
      <th>122904</th>
      <td>0.136612</td>
      <td>0.034962</td>
    </tr>
    <tr>
      <th>134130</th>
      <td>0.151885</td>
      <td>0.044751</td>
    </tr>
    <tr>
      <th>134853</th>
      <td>0.136332</td>
      <td>0.035934</td>
    </tr>
  </tbody>
</table>
<p>167 rows × 2 columns</p>
</div>




```python
rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]

```


```python
rec_percentages = rec_percentages.sort_values("score", ascending=False)
```


```python
rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>similar</th>
      <th>all</th>
      <th>score</th>
      <th>movieId</th>
      <th>title</th>
      <th>genres</th>
      <th>clean_title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4790</th>
      <td>1.000000</td>
      <td>0.047170</td>
      <td>21.200084</td>
      <td>4896</td>
      <td>Harry Potter and the Sorcerer's Stone (a.k.a. ...</td>
      <td>Adventure|Children|Fantasy</td>
      <td>Harry Potter and the Sorcerers Stone aka Harry...</td>
    </tr>
    <tr>
      <th>5704</th>
      <td>0.582738</td>
      <td>0.036641</td>
      <td>15.903887</td>
      <td>5816</td>
      <td>Harry Potter and the Chamber of Secrets (2002)</td>
      <td>Adventure|Fantasy</td>
      <td>Harry Potter and the Chamber of Secrets 2002</td>
    </tr>
    <tr>
      <th>11700</th>
      <td>0.408155</td>
      <td>0.029748</td>
      <td>13.720472</td>
      <td>54001</td>
      <td>Harry Potter and the Order of the Phoenix (2007)</td>
      <td>Adventure|Drama|Fantasy|IMAX</td>
      <td>Harry Potter and the Order of the Phoenix 2007</td>
    </tr>
    <tr>
      <th>10408</th>
      <td>0.506515</td>
      <td>0.037785</td>
      <td>13.405336</td>
      <td>40815</td>
      <td>Harry Potter and the Goblet of Fire (2005)</td>
      <td>Adventure|Fantasy|Thriller|IMAX</td>
      <td>Harry Potter and the Goblet of Fire 2005</td>
    </tr>
    <tr>
      <th>13512</th>
      <td>0.408435</td>
      <td>0.032491</td>
      <td>12.570839</td>
      <td>69844</td>
      <td>Harry Potter and the Half-Blood Prince (2009)</td>
      <td>Adventure|Fantasy|Mystery|Romance|IMAX</td>
      <td>Harry Potter and the HalfBlood Prince 2009</td>
    </tr>
    <tr>
      <th>7742</th>
      <td>0.582878</td>
      <td>0.048326</td>
      <td>12.061317</td>
      <td>8368</td>
      <td>Harry Potter and the Prisoner of Azkaban (2004)</td>
      <td>Adventure|Fantasy|IMAX</td>
      <td>Harry Potter and the Prisoner of Azkaban 2004</td>
    </tr>
    <tr>
      <th>15538</th>
      <td>0.394283</td>
      <td>0.034097</td>
      <td>11.563682</td>
      <td>81834</td>
      <td>Harry Potter and the Deathly Hallows: Part 1 (...</td>
      <td>Action|Adventure|Fantasy|IMAX</td>
      <td>Harry Potter and the Deathly Hallows Part 1 2010</td>
    </tr>
    <tr>
      <th>16718</th>
      <td>0.385316</td>
      <td>0.035471</td>
      <td>10.862722</td>
      <td>88125</td>
      <td>Harry Potter and the Deathly Hallows: Part 2 (...</td>
      <td>Action|Adventure|Drama|Fantasy|Mystery|IMAX</td>
      <td>Harry Potter and the Deathly Hallows Part 2 2011</td>
    </tr>
    <tr>
      <th>10450</th>
      <td>0.134090</td>
      <td>0.014553</td>
      <td>9.213661</td>
      <td>41566</td>
      <td>Chronicles of Narnia: The Lion, the Witch and ...</td>
      <td>Adventure|Children|Fantasy</td>
      <td>Chronicles of Narnia The Lion the Witch and th...</td>
    </tr>
    <tr>
      <th>11606</th>
      <td>0.124002</td>
      <td>0.014864</td>
      <td>8.342407</td>
      <td>53125</td>
      <td>Pirates of the Caribbean: At World's End (2007)</td>
      <td>Action|Adventure|Comedy|Fantasy</td>
      <td>Pirates of the Caribbean At Worlds End 2007</td>
    </tr>
  </tbody>
</table>
</div>




```python
def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > .10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]
```


```python
import ipywidgets as widgets
from IPython.display import display

movie_name_input = widgets.Text(
    value='Toy Story',
    description='Movie Title:',
    disabled=False
)
recommendation_list = widgets.Output()

def on_type(data):
    with recommendation_list:
        recommendation_list.clear_output()
        title = data["new"]
        if len(title) > 5:
            results = search(title)
            movie_id = results.iloc[0]["movieId"]
            display(find_similar_movies(movie_id))

movie_name_input.observe(on_type, names='value')

display(movie_name_input, recommendation_list)
```


    Text(value='Toy Story', description='Movie Title:')



    Output()



```python

```
