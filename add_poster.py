import requests
import pandas as pd
from tqdm import tqdm
import time
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_KEY = '43daab77ad55185339fd0bfdfe6e3f7c'  # 🔐 Replace with your TMDB API key
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
INPUT_CSV = 'data/Movies_dataset.csv'
OUTPUT_CSV = 'data/latest_movies_with_posters.csv'

# Setup session with retries
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

def get_poster_path(title):
    params = {
        'api_key': API_KEY,
        'query': title,
        'include_adult': False
    }
    try:
        response = session.get(SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            return data['results'][0].get('poster_path')
    except Exception as e:
        print(f"⚠️ Error fetching '{title}': {e}")
    return None

def main():
    # Load input
    df = pd.read_csv(INPUT_CSV)

    # Resume if file exists
    if os.path.exists(OUTPUT_CSV):
        print("🔁 Resuming from previous progress...")
        df_out = pd.read_csv(OUTPUT_CSV)
    else:
        df_out = df.copy()
        df_out['poster_path'] = None

    # Ensure we only work on rows without posters
    rows_to_update = df_out[df_out['poster_path'].isna()]

    print(f"🔍 Fetching poster paths for {len(rows_to_update)} movies...")
    for idx in tqdm(rows_to_update.index, total=len(rows_to_update)):
        title = df_out.at[idx, 'title']
        poster_path = get_poster_path(title)
        df_out.at[idx, 'poster_path'] = poster_path

        # Save progress every 50 rows
        if idx % 50 == 0:
            df_out.to_csv(OUTPUT_CSV, index=False)

        time.sleep(0.5)  # TMDB rate limit

    # Final save
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Done! Output saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
