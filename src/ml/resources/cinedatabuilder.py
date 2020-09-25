import operator
import shutil

import bs4
import ml.resources.anonsess
import re
import os
import pandas as pd
import time
import random
import simpleclock
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler("cinedatabuilder.log")
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

url_lookup = ""
url_reviews_template = ""
saved_data_path = os.path.join(os.path.dirname(__file__), "cine_critiques_presse.csv")
info_path = os.path.join(os.path.dirname(__file__), "cine_info.txt")

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


def process_review_html(html, movie_id, df_data):
    soup = bs4.BeautifulSoup(html, "html.parser")
    movie_title = extract_title(soup)
    review_tags = list(soup.find_all("div", id=RE_PRESSREVIEW))
    _n = len(review_tags)
    logger.info(f"found {_n} review(s)" if _n > 0 else "no review found...")
    stars_and_reviews = [(extract_note(tag), extract_text(tag)) for tag in review_tags]
    df = data_to_df(movie_id, movie_title, stars_and_reviews)
    return df_data.append(df)


def iter_urls(ns, **kwargs):
    for n in ns:
        resp = session.get(url_lookup.format(n=n))
        landmark = n
        yield from (map(lambda s: (url_reviews_template.format(movie_id=s),
            int(s), landmark), RE_IDS.findall(resp.text)) if resp.ok else [])

def sleep_duration():
    return random.randint(100, 1000) / 100


AUTOSAVE_EVERY = 10
BACKUP_EVERY = 50
PAGES = range(1, 100)

if __name__ == "__main__":
    clock = simpleclock.Clock.started()
    session = ml.resources.anonsess.get_session()

    df_saved_data = get_saved_data()
    df_data = empty_data()
    known_ids = set(df_saved_data["id"].values)
    autosave_counter = 0
    backup_counter = 0

    for url, url_id, landmark in iter_urls(PAGES):
        if url_id in known_ids:
            logger.info(f"{url_id} already in base")
        else:
            logger.info(f"processing {url_id} ({url})")
            known_ids.add(url_id)
            resp = session.get(url)
            if resp.ok:
                html = resp.text
                df_data = process_review_html(html, url_id, df_data)
                autosave_counter += 1
                backup_counter += 1
                if autosave_counter >= AUTOSAVE_EVERY:
                    logger.info("autosave...")
                    autosave_counter = 0
                    save_data(df_saved_data.append(df_data))
                    df_saved_data = get_saved_data()
                    df_data = empty_data()
                    with open(info_path, "w") as f:
                        f.write(f"last page: {landmark}")
                if backup_counter >= BACKUP_EVERY:
                    logger.info("backup...")
                    backup_counter = 0
                    backup_data()
            elif resp.status_code == 403:
                logger.info("leaving...")
                break
            else:
                logger.info(f"status code: {resp.status_code}")

            t_sleep = sleep_duration()
            logger.info(f"pause: {t_sleep:.2f}s")
            time.sleep(t_sleep)

    if autosave_counter > 0:
        logger.info("saving before exiting")
        save_data(df_saved_data.append(df_data))

    clock.elapsed_since_start.print("Total time")
