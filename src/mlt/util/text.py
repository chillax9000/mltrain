import unicodedata
from typing import Iterable


def remove_multispaces(s: str):
    return s.replace("  ", " ")


def get_sentences_from_text(text: str) -> Iterable:
    raw_sentences = text.split(".")
    sentences = map(str.strip,
                    map(remove_multispaces,
                        map(lambda s: s.replace("\n", " "), raw_sentences)))
    return sentences


def get_words_from_text(text: str) -> Iterable:
    current = ""
    for c in text:
        if c.isalpha():
            current += c
        elif current != "":
            yield current
            current = ""
    if current != "":
        yield current


def unicode_to_ascii(s: str):
    if 'œ' in s:
        s = s.replace('œ', 'oe')
    s_converted = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8')
    # if len(s) != len(s_converted):
    #     print("ascii conversion warning:", s, "->", s_converted)
    return s_converted
