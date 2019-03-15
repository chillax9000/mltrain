from typing import Iterable


def remove_multispaces(s: str):
    return s.replace("  ", " ")


def get_sentences_from_text(text: str) -> Iterable:
    raw_sentences = text.split(".")
    sentences = map(str.strip,
                    map(remove_multispaces,
                        map(lambda s: s.replace("\n", " "), raw_sentences)))
    return sentences
