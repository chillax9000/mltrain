"""
Using Morphalou - ATILF/ CNRS - Universités de Nancy 2.0: https://www.cnrtl.fr/lexiques/morphalou/
to build a vocabulary basis
"""

import os
import xml.etree.ElementTree
import time
import simpleclock

file_path = os.path.join(os.path.dirname(__file__), ".nocommit", "Morphalou-2.0.xml")

clock = simpleclock.Clock.started()

print("parsing xml")
tree = xml.etree.ElementTree.parse(file_path)
stop = time.perf_counter_ns()
clock.elapsed_since_start.print("end")

print("processing")
root = tree.getroot()
words = {}
for child in root:
    if child.tag == "lexicalEntry":
        lemmatized_form = child.find("formSet").find("lemmatizedForm")
        ortography = lemmatized_form.find("orthography").text
        grammatical_cat = lemmatized_form.find("grammaticalCategory").text
        words[ortography] = grammatical_cat
clock.elapsed_since_last_call.print("end")

grammatical_cats = set(words.values())
print(f"{len(words)} words saved")
print(f"grammatical categories found: {sorted(grammatical_cats)}")

with open(os.path.join(os.path.dirname(__file__), "vocab-fr.txt"), "w") as f:
    for word in words:
        f.write(f"{word}\n")
