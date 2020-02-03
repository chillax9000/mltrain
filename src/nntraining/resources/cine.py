import operator

import bs4
import requests
import re
import os
import pandas as pd


url_reviews_template = ""
saved_html_path = os.path.join(os.path.dirname(__file__), "ac_test_saved")
saved_data_path = os.path.join(os.path.dirname(__file__), "ac_saved_data")

COLUMNS = ["id", "titre", "note", "critique"]
RE_PRESSREVIEW = re.compile("pressreview")


def empty_data():
    return pd.DataFrame(columns=COLUMNS)


def get_saved_data():
    if os.path.exists(saved_data_path):
        return pd.read_csv(saved_data_path)

    return empty_data()


def save_data(df: pd.DataFrame):
    df.to_csv(saved_data_path, index=False)


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
    stars_and_reviews = [(extract_note(tag), extract_text(tag)) for tag in review_tags]
    df = data_to_df(movie_id, movie_title, stars_and_reviews)
    return df_data.append(df)


def get_movie_ids():
    return []


if __name__ == "__main__":
    df_saved_data = get_saved_data()
    ids = get_movie_ids()
    df_data = empty_data()
    for movie_id in ids:
        if movie_id in df_saved_data["id"].values:
            print(f"{movie_id} already in base")
        else:
            resp = requests.get(url_reviews_template.format(movie_id=movie_id))
            if resp.ok:
                html = resp.text
                df_data = process_review_html(html, df_data)

    save_data(df_saved_data.append(df_data))
