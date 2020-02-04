import operator
import shutil

import bs4
import nntraining.resources.anonsess
import re
import os
import pandas as pd
import time
import random
import simpleclock

url_lookup = ""
url_reviews_template = ""
saved_data_path = os.path.join(os.path.dirname(__file__), "cine_critiques_presse.csv")

RE_PRESSREVIEW = re.compile("pressreview")
RE_IDS = re.compile(r"([0-9]*).html")

COLUMNS = ["id", "titre", "note", "critique"]


def empty_data():
    return pd.DataFrame(columns=COLUMNS)


def get_saved_data():
    if os.path.exists(saved_data_path):
        return pd.read_csv(saved_data_path)

    return empty_data()


def save_data(df: pd.DataFrame):
    df.to_csv(saved_data_path, index=False)


def backup_data():
    shutil.copyfile(saved_data_path, os.path.join(saved_data_path + ".bckp"))


def data_to_df(movie_id, title, stars_and_reviews):
    n_reviews = len(stars_and_reviews)
    df = pd.DataFrame({
        "id": [movie_id] * n_reviews,
        "titre": [title] * n_reviews,
        "note": list(map(operator.itemgetter(0), stars_and_reviews)),
        "critique": list(map(operator.itemgetter(1), stars_and_reviews)),
    }, columns=COLUMNS)
    return df


def extract_title(soup):
    return soup.find("div", class_="titlebar").text


def extract_note(tag):
    t = tag.find("div", class_="stareval").find("div", class_="rating-mdl")
    note_str = t["class"][1]
    note = float(note_str[1:]) / 10
    return note


def extract_text(tag):
    return tag.find("p").text.strip()


def process_review_html(html, df_data):
    soup = bs4.BeautifulSoup(html, "html.parser")
    movie_title = extract_title(soup)
    print(f"processing movie: {movie_title}, id: {movie_id}")
    review_tags = list(soup.find_all("div", id=RE_PRESSREVIEW))
    if len(review_tags) == 0:
        print("no review found...")
    stars_and_reviews = [(extract_note(tag), extract_text(tag)) for tag in review_tags]
    df = data_to_df(movie_id, movie_title, stars_and_reviews)
    return df_data.append(df)


def get_movie_ids(n, session):
    resp = session.get(url_lookup.format(n=n))
    return map(int, RE_IDS.findall(resp.text)) if resp.ok else []


if __name__ == "__main__":
    clock = simpleclock.Clock.started()
    session = nntraining.resources.anonsess.get_session()

    df_saved_data = get_saved_data()
    df_data = empty_data()
    known_ids = set(df_saved_data["id"].values)
    AUTOSAVE_EVERY = 10
    BACKUP_EVERY = 50
    autosave_counter = 0
    backup_counter = 0

    for n in range(305, 500):
        id_batch = set(get_movie_ids(n, session))
        print(f"checking {len(id_batch)} movies, ({n})")
        for movie_id in id_batch:
            if movie_id in known_ids:
                print(f"{movie_id} already in base")
            else:
                known_ids.add(movie_id)
                resp = session.get(url_reviews_template.format(movie_id=movie_id))
                if resp.ok:
                    html = resp.text
                    df_data = process_review_html(html, df_data)
                    autosave_counter += 1
                    backup_counter += 1
                    if autosave_counter >= AUTOSAVE_EVERY:
                        print("autosave...")
                        autosave_counter = 0
                        save_data(df_saved_data.append(df_data))
                        df_saved_data = get_saved_data()
                        df_data = empty_data()
                    if backup_counter >= BACKUP_EVERY:
                        print("backup...")
                        backup_counter = 0
                        backup_data()
                elif resp.status_code == 403:
                    print("leaving...")
                    break
                else:
                    print(f"status code: {resp.status_code}")

                t_sleep = random.randint(200, 800) / 100
                print(f"waiting {t_sleep: .2f}s")
                time.sleep(t_sleep)
        print()

    if autosave_counter > 0:
        print("saving before exiting")
        save_data(df_saved_data.append(df_data))

    clock.elapsed_since_start.print("Total time")
