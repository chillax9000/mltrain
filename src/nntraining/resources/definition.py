import os
import re
import simpleclock

xml_path = os.path.join(".nocommit", "frwiktionary-20191020-pages-articles.xml")
raw_path = os.path.join(".nocommit", "raw")
pruned_path = os.path.join(".nocommit", "pruned")
refined_path = os.path.join(".nocommit", "refined")

MAX_C = 20_000_000

RE_XML_TO_RAW = re.compile(r"('''|===|# )")
RE_FIND_WORD = re.compile(r"^'''[^']*'''")
RE_FIND_LINK = re.compile(r"\[\[(?P<link>[^|\]]+)\|?(?P<text>[^|\]]+)?\]\]")
RE_FIND_CAT = re.compile(r"\{\{(.+?)\}\}")
RE_FIND_FICHIER = re.compile(r"\[\[(Fichier|File|Image):.*?\]\]")


def extract_and_replace_links(line):
    match = RE_FIND_LINK.search(line)
    links = []
    while match:
        start, end = match.start(), match.end()
        link, text = match.group("link"), match.group("text")
        text = text if text else link
        line = line[:start] + text + line[end:]
        links.append(link)
        match = RE_FIND_LINK.search(line)
    return line.strip(), links


def extract_and_delete_categories(line):
    match = RE_FIND_CAT.search(line)
    categories = []
    while match:
        start, end = match.start(), match.end()
        category = match.group(1)
        line = line[:start] + line[end:]
        categories.append(category)
        match = RE_FIND_CAT.search(line)
    return line.strip(), categories


def delete_fichiers(line):
    match = RE_FIND_FICHIER.search(line)
    while match:
        line = line[:match.start()] + line[match.end():]
        match = RE_FIND_FICHIER.search(line)
    return line


def check_start_def(line):
    return line.startswith("=== {{S") and line.endswith("fr}} ===\n")


def check_in_def(line, in_def):
    return (in_def and not line.startswith("===")) or check_start_def(line)


def process_def_line(line):
    matches = RE_FIND_WORD.findall(line)
    if matches:
        return matches[0][3:-3] + "\n"
    return line


def xml_to_raw(input_path, output_path, max_lines_w=MAX_C):
    with open(input_path, "r") as f_r, open(output_path, "w") as f_w:
        c = 0
        for line in f_r:
            if RE_XML_TO_RAW.match(line):
                f_w.write(line)
                c += 1
            if c > max_lines_w:
                break


def prune_raw(input_path, output_path, max_lines_w=MAX_C):
    with open(input_path, "r") as f_r, open(output_path, "w") as f_w:
        c = 0
        in_def = False
        for line in f_r:
            in_def = check_in_def(line, in_def)
            if in_def:
                f_w.write(line)
                c += 1
            if c > max_lines_w:
                break


def refine_pruned(input_path, output_path, max_lines_w=MAX_C):
    with open(input_path, "r") as f_r, open(output_path, "w") as f_w:
        c = 0
        # missed = 0
        for line in f_r:
            if line.startswith("#"):
                line_orig = line
                line = line[1:].strip()
                line, cats = extract_and_delete_categories(line)
                line, links = extract_and_replace_links(line)
                line = delete_fichiers(line)
                # if "[[" in line or "{{" in line:
                #     if missed < 100:
                #         print(line)
                #     missed += 1
                line = "# " + line.strip() + " " + str({"links": links, "categories": cats}) + "\n"
            if line.startswith("'''"):
                line = process_def_line(line)
            f_w.write(line)
            c += 1
            if c > max_lines_w:
                break


if __name__ == "__main__":
    clock = simpleclock.Clock.started()
    xml_to_raw(xml_path, raw_path)
    clock.elapsed_since_last_call.print("xml to raw")
    prune_raw(raw_path, pruned_path)
    clock.elapsed_since_last_call.print("pruned")
    refine_pruned(pruned_path, refined_path)
    clock.elapsed_since_last_call.print("refined")
