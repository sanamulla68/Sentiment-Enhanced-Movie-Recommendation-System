import requests
import pandas as pd
from tqdm import tqdm
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY = '43daab77ad55185339fd0bfdfe6e3f7c'  # 🔑 Replace this with your TMDB key
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
DETAILS_URL = "https://api.themoviedb.org/3/movie/{}"  # movie_id goes here

INPUT_CSV = 'data/latest_movies_with_posters.csv'
OUTPUT_CSV = 'data/latest_movies_with_genres.csv'

# Setup retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_genres(title):
    params = {
        'api_key': API_KEY,
        'query': title,
        'include_adult': False
    }
    try:
        search_resp = session.get(SEARCH_URL, params=params, timeout=10).json()
        if search_resp.get('results'):
            movie = search_resp['results'][0]
            movie_id = movie['id']

            detail_resp = session.get(DETAILS_URL.format(movie_id), params={'api_key': API_KEY}, timeout=10).json()
            genres = [genre['name'] for genre in detail_resp.get('genres', [])]
            return ", ".join(genres)
    except Exception as e:
        print(f"❌ Error fetching genres for '{title}': {e}")
    return None

def main():
    df = pd.read_csv(INPUT_CSV)

    # Check if genres already exist
    if 'genres' not in df.columns:
        df['genres'] = None

    print(f"🔍 Fetching genres for {len(df)} movies...")
    for idx in tqdm(df.index):
        if pd.isna(df.at[idx, 'genres']):
            title = df.at[idx, 'title']
            genre_str = get_genres(title)
            df.at[idx, 'genres'] = genre_str
            time.sleep(0.5)  # Respect TMDB rate limit

        # Save progress every 50 movies
        if idx % 50 == 0:
            df.to_csv(OUTPUT_CSV, index=False)

    df.to_csv(OUTPUT_CSV, index=False)
    print("✅ Genre fetching complete.")

if __name__ == "__main__":
    main()
