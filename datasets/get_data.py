import pandas
import os

def get_data(path_root, mode="train"):
  
  path = path_root
  if mode == "train":
    path = os.path.join(path, "movies_train.dat")
  else:
    path = os.path.join(path, "movies_test.dat")

  movies = pandas.read_csv(path, engine='python',
                         sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
  # movies['genre'] = ";".join(movies.genre.str.split('|'))

  folder_img_path = os.path.join(path_root, "ml1m-images")
  movies['id'] = movies.index
  movies.reset_index(inplace=True)
  movies['img_path'] = movies.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)
  return movies

def check(row):
  return not os.path.exists(row["img_path"])
 
def get_dataframe(path):
  data = pandas.read_csv(path)
  mask = data.apply(check, axis=1)

  # Remove rows based on the mask
  data = data.drop(data[mask].index)

  return data