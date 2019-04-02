import collections
import enum
import os
from typing import Tuple, Iterable
from nntraining.util.text import get_sentences_from_text

import bs4
import requests

from nntraining.clock import Clock

from nntraining.resources import base_path

import re

source_text_folder = os.path.join(base_path, "saved", "wiki")
cleaned_text_folder = os.path.join(source_text_folder, "cleaned")


class Lang(enum.Enum):
    fr = "fr"
    en = "en"

    def __init__(self, code):
        self.code = code

    @property
    def wiki_source_folder(self):
        return os.path.join(source_text_folder, self.code)

    @property
    def wiki_cleaned_text_folder(self):
        return os.path.join(cleaned_text_folder, self.code)

    def wiki_cleaned_text_file(self, name: str):
        return os.path.join(self.wiki_cleaned_text_folder, name)

    def iter_wiki_cleaned_text_file_paths(self):
        for name in os.listdir(self.wiki_cleaned_text_folder):
            yield os.path.join(self.wiki_cleaned_text_folder, name)


default_pages = {
    Lang.fr: ["Art", "Chat", "Chien", "France", "Histoire", "Jeu", "Langue", "Pays", "Ville", "Être_humain"],
    Lang.en: ["Art", "Cat", "Dog", "England", "History", "Game", "Language", "Country", "City", "Human_being"],
}

main_folders = [source_text_folder, cleaned_text_folder]
lang_folders = [folder for lang in Lang for folder in [lang.wiki_source_folder,
                                                       lang.wiki_cleaned_text_folder]]
folders = main_folders + lang_folders
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_html(word: str, lang: Lang) -> Tuple[str, str]:
    file_path = os.path.join(lang.wiki_source_folder, word)
    if os.path.exists(file_path):
        source = "disc"
        print("reading", file_path)
        with open(file_path) as f:
            html = f.read()
    else:
        source = "internet"
        print("downloading to", file_path)
        resp = requests.get(f"http://{lang.code}.wikipedia.org/wiki/{word}")
        html = resp.text
        with open(file_path, "w") as f:
            f.write(html)
    return html, source


def rm_footnotes_refs(text):
    return re.sub(r"\[[0-9]*\]", "", text)


def clean_text(text: str) -> str:
    text = rm_footnotes_refs(text)
    text = text.replace("'", "' ")
    return text


def save_cleaned_text(word: str, lang: Lang, paragraph_min_len=80):
    html, source = get_html(word, lang)
    soup = bs4.BeautifulSoup(html, "html.parser")  # slow
    file_path = os.path.join(lang.wiki_cleaned_text_folder, word)
    with open(file_path, "w") as f:
        for p in soup.find_all("p"):
            text = p.get_text()
            if len(text) > paragraph_min_len:
                text_cleaned = clean_text(text)
                f.write(text_cleaned)


def save_cleaned_all(lang: Lang):
    for name in os.listdir(lang.wiki_source_folder):
        save_cleaned_text(name, lang)


def get_cleaned_text(word: str, lang: Lang):
    file_path = os.path.join(lang.wiki_cleaned_text_folder, word)
    if not os.path.exists(file_path):
        save_cleaned_text(word, lang)
    with open(file_path) as f:
        return f.read()


def get_words_dict(wikipage: str = "France", lang: Lang = Lang.fr, paragraph_min_len=80):
    clock = Clock()
    clock.start()
    html, source = get_html(wikipage, lang)
    clock.print_elapsed_since_last_call(f"getting html from {source}")

    soup = bs4.BeautifulSoup(html, "html.parser")  # slow
    clock.print_elapsed_since_last_call("parsing html")

    word_counter = collections.Counter()
    for p in soup.find_all("p"):
        text = p.get_text()
        if len(text) > paragraph_min_len:
            text_cleaned = clean_text(text)
            for word in text_cleaned.split():
                word_counter[word] += 1

    to_del = []
    for word in word_counter:
        if not any(c.isalpha() for c in word):
            to_del.append(word)
    for word in to_del:
        del word_counter[word]
    clock.print_elapsed_since_last_call("parsing html")

    return word_counter


def get_all_chars(lang: Lang):
    chars = set()
    folder = lang.wiki_cleaned_text_folder
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name)) as f:
            _chars = set(f.read())
            chars.update(_chars)
    return chars


def get_sentences_list_from_file(lang: Lang, name: str) -> Iterable:
    with open(lang.wiki_cleaned_text_file(name)) as f:
        text = f.read()
    return get_sentences_from_text(text)


def get_sentences_list(lang: Lang):
    sentences = []
    for name in os.listdir(lang.wiki_cleaned_text_folder):
        sentences += get_sentences_list_from_file(lang, name)
    return sentences
