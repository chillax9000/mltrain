"""
Using Morphalou - ATILF/ CNRS - Universités de Nancy 2.0: https://www.cnrtl.fr/lexiques/morphalou/
to build a vocabulary basis
"""

import os
import xml.etree.ElementTree
import time
import simpleclock
from collections import defaultdict

file_path = os.path.join(os.path.dirname(__file__), ".nocommit", "Morphalou-2.0.xml")

clock = simpleclock.Clock.started()

tree = xml.etree.ElementTree.parse(file_path)
stop = time.perf_counter_ns()
clock.elapsed_since_start.print("parsing took")

root = tree.getroot()
words = defaultdict(lambda: set())
for lexical_entry in root.findall("lexicalEntry"):
    formset = lexical_entry.find("formSet")
    # base
    lemmatized_form = formset.find("lemmatizedForm")
    orthography = lemmatized_form.find("orthography").text
    grammatical_cat = lemmatized_form.find("grammaticalCategory").text
    words[orthography].add(grammatical_cat)
    # inflections
    for inflected_form in formset.iterfind("inflectedForm"):
        orthography = inflected_form.find("orthography").text
        words[orthography].add(grammatical_cat)
clock.elapsed_since_last_call.print("processing took")

grammatical_cats = set()
for cats in words.values():
    grammatical_cats.update(cats)
print(f"{len(words)} words saved")
print(f"grammatical categories found: {sorted(grammatical_cats)}")

with open(os.path.join(os.path.dirname(__file__), "vocab-fr.txt"), "w") as f:
    for word in words:
        f.write(f"{word}\n")
clock.elapsed_since_last_call.print("writing took")

"""
 <lexicalEntry id="abandonner_1">
                <formSet>
                        <lemmatizedForm>
                                <orthography>abandonner</orthography>
                                <grammaticalCategory>verb</grammaticalCategory>
                        </lemmatizedForm>
                        <inflectedForm>
                                <orthography>abandonna</orthography>
                                <grammaticalNumber>singular</grammaticalNumber>
                                <grammaticalMood>indicative</grammaticalMood>
                                <grammaticalTense>simplePast</grammaticalTense>
                                <grammaticalPerson>thirdPerson</grammaticalPerson>
                        </inflectedForm>
"""