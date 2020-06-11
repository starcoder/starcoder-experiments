import argparse
import pandas
import gzip
import json
import re

def format_stress(s):
    m = re.match(r"^([SuxeE]+?)(ele?)?$", str(s))
    nonfinal, elision = m.groups() if m else ("", None)
    vals = tuple(list(nonfinal) + ([elision] if elision else []))
    return vals

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    same_stress, total_stress = 0, 0
    corrections = {}
    corrected_lines = set()
    sentences = []
    
    data = pandas.read_excel(args.inputs[0])

    # these are the lines that have been annotated
    training_offsets = list(range(859)) + list(range(17452, 18300))
    all_offsets = list(range(25000))
    entities = {"canterbury" : {"entity_type" : "book"}}
    section_counts = {}
    for offset in all_offsets:
        filename, section, _, _ = data.iloc[offset * 4, 0:4]
        section_counts[section] = section_counts.get(section, 0) + 1
        line_num = section_counts[section]
        item = data.iloc[offset * 4 : (offset + 1) * 4, 4:].T
        line_id = "canterbury_{}_{}".format(section, line_num)
        entities[section] = {"entity_type" : "section",
                             "section_from" : "canterbury"}
        entities[line_id] = {"entity_type" : "line",                                 
                             "line_from" : section}
        c = list(item.columns)[0]
        trimmed = item[item[c].replace({float("NaN"): False}) != False]
        for token_num in range(1, trimmed.shape[0] + 1):
            token, prosodic_stress, corrected_stress, lexical_info = trimmed.iloc[token_num - 1]
            if token.startswith("CC") or token.startswith("CHK") or "ok?" in token:
                continue
            try:
                lemma, tagging = re.match(r"^\s*(.*)\@(.*)\s*$", str(lexical_info)).groups()
            except:
                print(token, lexical_info)
            tags = tagging.split("%")                
            total_stress += 1
            prosodic_stress = format_stress(prosodic_stress)
            corrected_stress = format_stress(corrected_stress)
            word_id = "{}_{}".format(line_id, token_num)
            entities[word_id] = {"entity_type" : "word",
                                 "token" : token,
                                 "lemma" : lemma,
                                 "tagging" : tagging,
                                 "stress" : "".join(corrected_stress),
                                 "word_from" : line_id
                                 }
    with gzip.open(args.output, "wt") as ofd:
        for eid, entity in entities.items():
            entity["id"] = eid            
            ofd.write(json.dumps(entity) + "\n")
