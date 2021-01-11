#!/usr/bin/env python
# coding: utf-8

# # Left join
# 
# ```python
# 
# # Merge with left join
# movies_taglines = movies.merge(taglines, on='id', how='left')
# 
# ```

# [Counting missing rows with left join | Python](https://campus.datacamp.com/courses/joining-data-with-pandas/merging-tables-with-different-join-types?ex=2)
# 
# > ## Counting missing rows with left join
# > 
# > The Movie Database is supported by volunteers going out into the world, collecting data, and entering it into the database. This includes financial data, such as movie budget and revenue. If you wanted to know which movies are still missing data, you could use a left join to identify them. Practice using a left join by merging the `movies` table and the `financials` table.
# > 
# > The `movies` and `financials` tables have been loaded for you.

# ### init

# In[1]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(movies , financials)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'financials.csv': 'https://file.io/URfUB8YN5BW1',
  'movies.csv': 'https://file.io/Crz2lkLMMkM6'}}
"""
prefixToc='1.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
movies = pd.read_csv(prefix+'movies.csv',index_col=0)
financials = pd.read_csv(prefix+'financials.csv',index_col=0)


# ### code

# > What column is likely the best column to merge the two tables on?

# In[2]:


movies.head()


# In[3]:


financials.head()


# > -   Merge the `movies` table, as the left table, with the `financials` table using a left join, and save the result to `movies_financials`.

# In[4]:


# Merge movies and financials with a left join
movies_financials = movies.merge(financials, on='id', how='left')


# > Count the number of rows in `movies_financials` with a null value in the `budget` column.

# In[5]:


# Count the number of rows in the budget column that are missing
number_of_missing_fin = movies_financials['budget'].isna().sum()

# Print the number of movies missing financials
print(number_of_missing_fin)


# [Enriching a dataset | Python](https://campus.datacamp.com/courses/joining-data-with-pandas/merging-tables-with-different-join-types?ex=3)
# 
# > ## Enriching a dataset
# > 
# > Setting `how='left'` with the `.merge()`method is a useful technique for enriching or enhancing a dataset with additional information from a different table. In this exercise, you will start off with a sample of movie data from the movie series _Toy Story_. Your goal is to enrich this data by adding the marketing tag line for each movie. You will compare the results of a left join versus an inner join.
# > 
# > The `toy_story` DataFrame contains the _Toy Story_ movies. The `toy_story` and `taglines` DataFrames have been loaded for you.

# ### init

# In[6]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(toy_story , taglines)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'taglines.csv': 'https://file.io/QRJBFlQh4D8V',
  'toy_story.csv': 'https://file.io/GlOCxyv3kref'}}
"""
prefixToc='1.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
toy_story = pd.read_csv(prefix+'toy_story.csv',index_col=0)
taglines = pd.read_csv(prefix+'taglines.csv',index_col=0)


# ### code

# > Merge `toy_story` and `taglines` on the `id` column with a **left join**, and save the result as `toystory_tag`.

# In[7]:


# Merge the toy_story and taglines tables with a left join
toystory_tag = toy_story.merge(taglines, on='id', how='left')

# Print the rows and shape of toystory_tag
print(toystory_tag)
print(toystory_tag.shape)


# > -   With `toy_story` as the left table, merge to it `taglines` on the `id` column with an **inner join**, and save as `toystory_tag`.

# In[8]:


# Merge the toy_story and taglines tables with a inner join
toystory_tag = toy_story.merge(taglines, on='id', how='inner')

# Print the rows and shape of toystory_tag
print(toystory_tag)
print(toystory_tag.shape)


# [How many rows with a left join? | Python](https://campus.datacamp.com/courses/joining-data-with-pandas/merging-tables-with-different-join-types?ex=4)
# 
# > ## How many rows with a left join?
# > 
# > Select the **true** statement about left joins.
# > 
# > Try running the following code statements in the console.
# > 
# > -   `left_table.merge(one_to_one, on='id', how='left').shape`
# > -   `left_table.merge(one_to_many, on='id', how='left').shape`
# > 
# > Note that the `left_table` starts out with **4** rows.

# ### init

# In[9]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(left_table , one_to_one, one_to_many)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'left_table.csv': 'https://file.io/VtIxKrUqOM1X',
  'one_to_many.csv': 'https://file.io/ADYv4xdWwVvc',
  'one_to_one.csv': 'https://file.io/MlxF2y4g5RTk'}}
"""
prefixToc='1.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
left_table = pd.read_csv(prefix+'left_table.csv',index_col=0)
one_to_one = pd.read_csv(prefix+'one_to_one.csv',index_col=0)
one_to_many = pd.read_csv(prefix+'one_to_many.csv',index_col=0)


# ### code

# In[10]:


left_table.merge(one_to_one, on='id', how='left').shape


# In[11]:


left_table.merge(one_to_many, on='id', how='left').shape


# In[12]:


one_to_one.head()


# In[13]:


one_to_many.head()


# In[14]:


left_table.head()


# ![image.png](attachment:image.png)

# # Other joins
# 
# ```python
# 
# # Merge with right join
# tv_movies = movies.merge(tv_genre, how='right',
# left_on='id', right_on='movie_id')
# 
# # Merge with outer join
# family_comedy = family.merge(comedy, on='movie_id', how='outer',
# suffixes=('_fam', '_com'))
# ```

# [Right join to find unique movies | Python](https://campus.datacamp.com/courses/joining-data-with-pandas/merging-tables-with-different-join-types?ex=6)
# 
# > ## Right join to find unique movies
# > 
# > Most of the recent big-budget science fiction movies can also be classified as action movies. You are given a table of science fiction movies called `scifi_movies` and another table of action movies called `action_movies`. Your goal is to find which movies are considered only science fiction movies. Once you have this table, you can merge the `movies` table in to see the movie names. Since this exercise is related to science fiction movies, use a right join as your superhero power to solve this problem.
# > 
# > The `movies`, `scifi_movies`, and `action_movies` tables have been loaded for you.

# ### init

# In[15]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(movies, scifi_movies,  action_movies)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'action_movies.csv': 'https://file.io/aqmekUv46CrP',
  'movies.csv': 'https://file.io/l29gpSVb4HAj',
  'scifi_movies.csv': 'https://file.io/4mcXpohFDl9w'}}
"""
prefixToc='2.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
movies = pd.read_csv(prefix+'movies.csv',index_col=0)
scifi_movies = pd.read_csv(prefix+'scifi_movies.csv',index_col=0)
action_movies = pd.read_csv(prefix+'action_movies.csv',index_col=0)


# ### code

# > Merge `action_movies` and `scifi_movies` tables with a **right join** on `movie_id`. Save the result as `action_scifi`.

# In[19]:


# Merge action_movies to scifi_movies with right join
action_scifi = action_movies.merge(scifi_movies, how='right', on='movie_id')


# > Update the merge to add suffixes, where `'_act'` and `'_sci'` are suffixes for the left and right tables, respectively.

# In[21]:


# Merge action_movies to scifi_movies with right join
action_scifi = action_movies.merge(scifi_movies, on='movie_id', how='right',
                                   suffixes=['_act', '_sci'])

# Print the first few rows of action_scifi to see the structure
print(action_scifi.head())


# > From `action_scifi`, subset only the rows where the `genre_act` column is null.

# In[22]:


# From action_scifi, select only the rows where the genre_act column is null
scifi_only = action_scifi[action_scifi['genre_act'].isna()]


# > -   Merge `movies` and `scifi_only` using the `id` column in the left table and the `movie_id` column in the right table with an inner join.

# In[25]:


# Merge the movies and scifi_only tables with an inner join
movies_and_scifi_only = movies.merge(scifi_only, left_on='id', right_on='movie_id')

# Print the first few rows and shape of movies_and_scifi_only
print(movies_and_scifi_only.head())
print(movies_and_scifi_only.shape)


# [Popular genres with right join | Python](https://campus.datacamp.com/courses/joining-data-with-pandas/merging-tables-with-different-join-types?ex=7)
# 
# > ## Popular genres with right join
# > 
# > What are the genres of the most popular movies? To answer this question, you need to merge data from the `movies` and `movie_to_genres` tables. In a table called `pop_movies`, the top 10 most popular movies in the `movies` table have been selected. To ensure that you are analyzing all of the popular movies, merge it with the `movie_to_genres` table using a right join. To complete your analysis, count the number of different genres. Also, the two tables can be merged by the movie ID. However, in `pop_movies` that column is called `id`, and in `movies_to_genres` it's called `movie_id`.
# > 
# > The `pop_movies` and `movie_to_genres` tables have been loaded for you.

# ### init

# In[26]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(pop_movies , movie_to_genres)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'movie_to_genres.csv': 'https://file.io/cgjWx1iJkE49',
  'pop_movies.csv': 'https://file.io/Oncd7QhBhwX7'}}
"""
prefixToc='2.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
pop_movies = pd.read_csv(prefix+'pop_movies.csv',index_col=0)
movie_to_genres = pd.read_csv(prefix+'movie_to_genres.csv',index_col=0)


# ### code

# > -   Merge `movie_to_genres` and `pop_movies` using a right join. Save the results as `genres_movies`.
# > -   Group `genres_movies` by `genre` and count the number of `id` values.

# In[28]:


import matplotlib.pyplot as plt


# In[29]:


# Use right join to merge the movie_to_genres and pop_movies tables
genres_movies = movie_to_genres.merge(pop_movies, how='right', 
                                      left_on='movie_id', 
                                      right_on='id')

# Count the number of genres
genre_count = genres_movies.groupby('genre').agg({'id':'count'})

# Plot a bar chart of the genre_count
genre_count.plot(kind='bar')
plt.show()


# [Using outer join to select actors | Python](https://campus.datacamp.com/courses/joining-data-with-pandas/merging-tables-with-different-join-types?ex=8)
# 
# > ## Using outer join to select actors
# > 
# > One cool aspect of using an outer join is that, because it returns all rows from both merged tables and null where they do not match, you can use it to find rows that do not have a match in the other table. To try for yourself, you have been given two tables with a list of actors from two popular movies: _Iron Man 1_ and _Iron Man 2_. Most of the actors played in both movies. Use an outer join to find actors who **_did not_** act in both movies.
# > 
# > The _Iron Man 1_ table is called `iron_1_actors`, and _Iron Man 2_ table is called `iron_2_actors`. Both tables have been loaded for you and a few rows printed so you can see the structure.
# ![image.png](attachment:image.png)

# ### init

# In[30]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(iron_1_actors , iron_2_actors)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'iron_1_actors.csv': 'https://file.io/SShwYkusjLve',
  'iron_2_actors.csv': 'https://file.io/wpAZabxCdjGO'}}
"""
prefixToc='2.3'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
iron_1_actors = pd.read_csv(prefix+'iron_1_actors.csv',index_col=0)
iron_2_actors = pd.read_csv(prefix+'iron_2_actors.csv',index_col=0)


# ### code

# > -   Save to `iron_1_and_2` the merge of `iron_1_actors` (left) with `iron_2_actors` tables with an outer join on the `id` column, and set suffixes to `('_1','_2')`.
# > -   Create an index that returns `True` if `name_1` or `name_2` are null, and `False` otherwise.

# In[32]:


# Merge iron_1_actors to iron_2_actors on id with outer join using suffixes
iron_1_and_2 = iron_1_actors.merge(iron_2_actors,
                                     on='id',
                                     how='outer',
                                     suffixes=['_1', '_2'])

# Create an index that returns true if name_1 or name_2 are null
m = ((iron_1_and_2['name_1'].isna()) | 
     (iron_1_and_2['name_2'].isna()))

# Print the first few rows of iron_1_and_2
print(iron_1_and_2[m].head())


# # Merging a table to itself
# 
# ```python
# 
# # Merging a table to itself
# original_sequels = sequels.merge(sequels, left_on='sequel', right_on='id',
# suffixes=('_org','_seq'))
# 
# ```

# [Self join | Python](https://campus.datacamp.com/courses/joining-data-with-pandas/merging-tables-with-different-join-types?ex=10)
# 
# > ## Self join
# > 
# > Merging a table to itself can be useful when you want to compare values in a column to other values in the same column. In this exercise, you will practice this by creating a table that for each movie will list the movie director and a member of the crew on one row. You have been given a table called `crews`, which has columns `id`, `job`, and `name`. First, merge the table to itself using the movie ID. This merge will give you a larger table where for each movie, every job is matched against each other. Then select only those rows with a director in the left table, and avoid having a row where the director's job is listed in both the left and right tables. This filtering will remove job combinations that aren't with the director.
# > 
# > The `crews` table has been loaded for you.

# ### init

# In[33]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(crews)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'crews.csv': 'https://file.io/80RSaOa3YlhE'}}
"""
prefixToc='3.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
crews = pd.read_csv(prefix+'crews.csv',index_col=0)


# ### code

# In[34]:


crews.head()


# > To a variable called `crews_self_merged`, merge the `crews` table to itself on the `id` column using an inner join, setting the suffixes to `'_dir'` and `'_crew'` for the left and right tables respectively.

# In[35]:


# Merge the crews table to itself
crews_self_merged = crews.merge(crews, on='id', suffixes=['_dir', '_crew'])


# In[36]:


crews_self_merged.head()


# > Create a Boolean index, named `boolean_filter`, that selects rows from the left table with the _job_ of `'Director'` and avoids rows with the _job_ of `'Director'` in the right table.

# In[37]:


# Create a Boolean index to select the appropriate
boolean_filter = ((crews_self_merged['job_dir'] == 'Director') & 
     (crews_self_merged['job_crew'] != 'Director'))
direct_crews = crews_self_merged[boolean_filter]


# In[38]:


direct_crews.head()


# > Use the `.head()` method to print the first few rows of `direct_crews`.

# In[39]:


# Print the first few rows of direct_crews
print(direct_crews.head())


# # Merging on indexes
# 
# ```python
# 
# # Setting an index
# movies = pd.read_csv('tmdb_movies.csv', index_col=['id'])
# 
# # Merging on index
# movies_taglines = movies.merge(taglines, on='id', how='left')
# 
# # MultiIndex merge
# samuel_casts = samuel.merge(casts, on=['movie_id','cast_id'])
# 
# # Index merge with left_on and right_on
# movies_genres = movies.merge(movie_to_genres, left_on='id', left_index=True,
# right_on='movie_id', right_index=True)
# ```

# [Index merge for movie ratings | Python](https://campus.datacamp.com/courses/joining-data-with-pandas/merging-tables-with-different-join-types?ex=13)
# 
# > ## Index merge for movie ratings
# > 
# > To practice merging on indexes, you will merge `movies` and a table called `ratings` that holds info about movie ratings. Make sure your merge returns **all** of the rows from the `movies` table and not all the rows of `ratings` table need to be included in the result.
# > 
# > The `movies` and `ratings` tables have been loaded for you.

# ### init
# 

# In[40]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(movies , ratings)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'movies.csv': 'https://file.io/wSWt80u8FPmk',
  'ratings.csv': 'https://file.io/MO6LpmMIVaLb'}}
"""
prefixToc='4.1'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
movies = pd.read_csv(prefix+'movies.csv',index_col=0)
ratings = pd.read_csv(prefix+'ratings.csv',index_col=0)


# ### code

# > Merge `movies` and `ratings` on the index and save to a variable called `movies_ratings`, ensuring that all of the rows from the `movies` table are returned.

# In[42]:


# Merge to the movies table the ratings table on the index
movies_ratings = movies.merge(ratings, on='id')

# Print the first few rows of movies_ratings
print(movies_ratings.head())


# [Do sequels earn more? | Python](https://campus.datacamp.com/courses/joining-data-with-pandas/merging-tables-with-different-join-types?ex=14)
# 
# > ## Do sequels earn more?
# > 
# > It is time to put together many of the aspects that you have learned in this chapter. In this exercise, you'll find out which movie sequels earned the most compared to the original movie. To answer this question, you will merge a modified version of the `sequels` and `financials` tables where their index is the movie ID. You will need to choose a merge type that will return all of the rows from the `sequels` table and not all the rows of `financials` table need to be included in the result. From there, you will join the resulting table to itself so that you can compare the revenue values of the original movie to the sequel. Next, you will calculate the difference between the two revenues and sort the resulting dataset.
# > 
# > The `sequels` and `financials` tables have been provided.

# ### init

# In[43]:


###################
##### Dataframe
###################

#upload and download

from downloadfromFileIO import saveFromFileIO
""" à executer sur datacamp: (apres copie du code uploadfromdatacamp.py)
uploadToFileIO(sequels , financials)

"""

tobedownloaded="""
{pandas.core.frame.DataFrame: {'financials.csv': 'https://file.io/PfTWnRDG6NGD',
  'sequels.csv': 'https://file.io/O1FAwt85wa5a'}}
"""
prefixToc='4.2'
prefix = saveFromFileIO(tobedownloaded, prefixToc=prefixToc, proxy="")

#initialisation

import pandas as pd
sequels = pd.read_csv(prefix+'sequels.csv',index_col=0)
financials = pd.read_csv(prefix+'financials.csv',index_col=0)


# ### code

# > -   With the `sequels` table on the left, merge to it the `financials` table on index named `id`, ensuring that all the rows from the `sequels` are returned and some rows from the other table may not be returned, Save the results to `sequels_fin`.

# In[44]:


sequels.head()


# In[45]:


financials.head()


# In[47]:


# Merge sequels and financials on index id
sequels_fin = sequels.merge(financials, on='id', how='left')


# In[48]:


sequels_fin.head()


# > Merge the `sequels_fin` table to itself with an inner join, where the left and right tables merge on `sequel` and `id` respectively with suffixes equal to `('_org','_seq')`, saving to `orig_seq`.

# In[49]:


# Self merge with suffixes as inner join with left on sequel and right on id
orig_seq = sequels_fin.merge(sequels_fin, how='inner', left_on='sequel', 
                             right_on='id', right_index=True,
                             suffixes=['_org', '_seq'])

# Add calculation to subtract revenue_org from revenue_seq 
orig_seq['diff'] = orig_seq['revenue_seq'] - orig_seq['revenue_org']


# In[50]:


orig_seq.head()


# > Select the `title_org`, `title_seq`, and `diff` columns of `orig_seq` and save this as `titles_diff`.

# In[51]:


# Select the title_org, title_seq, and diff 
titles_diff = orig_seq[['title_org', 'title_seq', 'diff']]


# In[52]:


titles_diff.head()


# > -   Sort by `titles_diff` by `diff` in descending order and print the first few rows.

# In[53]:


# Print the first rows of the sorted titles_diff
print(titles_diff.sort_values('diff', ascending=False).head())


# In[ ]:




