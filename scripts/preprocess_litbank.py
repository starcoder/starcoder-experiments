import json
import gzip
import csv
import argparse
from glob import glob
import os.path
import re

annotations = ["entities", "quotations", "events", "coref"]

mapping = {"B-LOC" : "location",
           "B-GPE" : "gpe",
           "B-FAC" : "facility",
           "B-PER" : "person",
           "B-ORG" : "organization",
           "B-VEH" : "vehicle",
}

def get_words(ssnum, swnum, esnum, ewnum, lengths, tid):    
    start = (ssnum, swnum)
    end = (esnum, ewnum)
    retval = []
    for i in range(ssnum, esnum + 1):
        for j in range(1, lengths[(tid, i)] + 1):
            widx = (i, j)            
            if widx >= start and widx <= end:
                retval.append((i, j))        
    return retval

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="inputs", help="Input files")
    parser.add_argument("--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    entities = {entity_type : {} for entity_type in ["author", "year", "book", "sentence", "word",
                                                     "named_entity", "mention", "coreference", "copula", "appositive",
                                                     "event", "quote", "discourse_entity"]}
    #"person", "gpe", "facility", "organization", "location", "vehicle", "quote", "event", "copula", "appositive", "mention", "coreference"]}
    data = {}
    texts = {}
    lengths = {}
    
    with open(os.path.join(args.inputs[0], "README.md"), "rt") as ifd:
        for row in re.match(r".*(\|Gutenberg.*?\|)\n\s*\n.*", ifd.read(), re.M|re.S).group(1).split("\n")[2:]:
            tid, year, author, title = row.strip("|").split("|")
            #if not tid.startswith("9"):
            #    continue
            last, first = re.match(r"^(.*?), (.*)$", author).groups()
            author_id = "author {}".format(author)
            book_id = "book {}".format(tid)
            year_id = "year {}".format(year)
            entities["author"][author_id] = {"given_name" : first, "surname" : last}
            entities["book"][book_id] = {"title" : title, "written_by" : author_id, "published_in" : year_id}
            entities["year"][year_id] = {"anno_domini" : int(year)}
    
    for annotation_type in annotations:
        for ann_fname in glob(os.path.join(args.inputs[0], annotation_type, "tsv", "*[n,v]")):
            txt_fname = "{}.txt".format(os.path.splitext(ann_fname)[0])
            tid, book_title = re.match(r"(\d+)_(.*)_brat.txt", os.path.basename(txt_fname)).groups()
            #if not tid.startswith("9"):
            #    continue
            book_id = "book {}".format(tid)
            #assert book_id in entities["book"]            
            if annotation_type == "entities":
                cur_sentence_id = None
                with open(ann_fname, "rt") as ifd:
                    sentence_num = 1
                    cur_sentence_id = "sentence {} {}".format(tid, sentence_num)
                    entities["sentence"][cur_sentence_id] = {"from_book" : book_id}
                    word_num = 1
                    for line in ifd:
                        if re.match(r"^\s*$", line):
                            sentence_num += 1
                            cur_sentence_id = "sentence {} {}".format(book_id, sentence_num)                            
                            entities["sentence"][cur_sentence_id] = {"from_book" : book_id}
                            word_num = 0
                        else:
                            toks = line.split("\t")
                            word, etype = toks[0:2]
                            word_id = "word {} {} {}".format(tid, sentence_num, word_num)
                            lengths[(tid, sentence_num)] = word_num
                            word_entity = {"form" : word, "from_sentence" : cur_sentence_id}
                            entities["word"][word_id] = word_entity
                        word_num += 1
            elif annotation_type == "coref":
                with open(ann_fname, "rt") as ifd:                
                    for line in ifd:
                        toks = line.strip().split("\t")
                        if toks[0] == "APPOS":
                            _, a_mention_id, b_mention_id = toks
                            appositive_id = "appositive {} {} {}".format(tid, a_mention_id, b_mention_id)
                            a_mention_id = "mention {} {}".format(tid, a_mention_id)
                            b_mention_id = "mention {} {}".format(tid, b_mention_id)
                            entities["appositive"][appositive_id] = {"renames" : a_mention_id, "renames_to" : b_mention_id}
                        elif toks[0] == "COP":
                            _, object_mention_id, attribute_mention_id = toks
                            copula_id = "copula {} {} {}".format(tid, object_mention_id, attribute_mention_id)
                            object_mention_id = "mention {} {}".format(tid, object_mention_id)
                            attribute_mention_id = "mention {} {}".format(tid, attribute_mention_id)
                            entities["copula"][copula_id] = {"copula_object" : object_mention_id, "copula_attribute" : attribute_mention_id}
                        elif toks[0] == "COREF":
                            _, mention_id, discourse_entity_id = toks
                            coref_id = "coreference {} {} {}".format(tid, mention_id, discourse_entity_id)
                            mention_id = "mention {} {}".format(tid, mention_id)
                            discourse_entity_id = "discourse_entity {} {}".format(tid, discourse_entity_id)
                            entities["discourse_entity"][discourse_entity_id] = entities.get(discourse_entity_id,
                                                                                             {"id" : discourse_entity_id})
                            entities["coreference"][coref_id] = {"realized_by_mention" : mention_id,
                                                                 "realizes_discourse_entity" : discourse_entity_id}
                        elif toks[0] == "MENTION":
                            _, mention_id, ssnum, swnum, esnum, ewnum, form, entity_subtype, mention_type = toks
                            mention_id = "mention {} {}".format(tid, mention_id)
                            entities["mention"][mention_id] = {"mention_type" : mention_type, "named_entity_type" : entity_subtype}
                            ssnum, swnum, esnum, ewnum = map(int, [ssnum, swnum, esnum, ewnum])                            
                            for snum, wnum in get_words(*map(int, [ssnum + 1, swnum + 1, esnum + 1, ewnum + 1]), lengths, tid):
                                word_id = "word {} {} {}".format(tid, snum, wnum)
                                entities["word"][word_id]["part_of_mention"] = mention_id
            elif annotation_type == "events":
                with open(ann_fname, "rt") as ifd:
                    sentence_num = 1
                    word_num = 1
                    cur_event = {}
                    for line in ifd:
                        if re.match(r"^\s*$", line):
                            sentence_num += 1
                            cur_sentence_id = "sentence {} {}".format(tid, sentence_num)
                            word_num = 0
                        else:
                            toks = line.split("\t")
                            word, etype = toks[0:2]
                            word_id = "word {} {} {}".format(tid, sentence_num, word_num)
                            if etype.startswith("EVENT"):
                                if cur_event == {}:
                                    cur_event = {"id" : "event {} {} {}".format(tid, sentence_num, word_num),
                                                 #"event_span" : [],
                                                 "entity_type" : "event"}
                                #cur_event["event_span"].append(word)
                                entities["word"][word_id]["realizes_event"] = cur_event["id"]
                            elif etype.startswith("O"):
                                if cur_event != {}:
                                    #cur_event["event_span"] = " ".join(cur_event["event_span"])
                                    entities["event"][cur_event["id"]] = cur_event
                                cur_event = {}
                        word_num += 1
            elif annotation_type == "quotations":
                with open(ann_fname, "rt") as ifd:
                    for line in ifd:
                        toks = line.strip().split("\t")
                        if toks[0] == "QUOTE":
                            _, quote_id, ssnum, swnum, esnum, ewnum, span = toks
                            quote_id = "quote {} {}".format(tid, quote_id)
                            entities["quote"][quote_id] = {}                            
                            ssnum, swnum, esnum, ewnum = map(int, [ssnum, swnum, esnum, ewnum])
                            for snum, wnum in get_words(*map(int, [ssnum + 1, swnum + 1, esnum + 1, ewnum + 1]), lengths, tid):
                                word_id = "word {} {} {}".format(tid, snum, wnum)
                                entities["word"][word_id]["part_of_quote"] = quote_id
                        elif toks[0] == "ATTRIB":
                            _, quote_id, speaker_id = toks
                            quote_id = "quote {} {}".format(tid, quote_id)
                            speaker_id = "discourse_entity {} {}".format(tid, speaker_id)
                            entities["quote"][quote_id]["spoken_by"] = speaker_id
                            entities["discourse_entity"][speaker_id] = entities["discourse_entity"].get(speaker_id, {})

    with gzip.open(args.output, "wt") as ofd:
        for entity_type, entries in entities.items():
            for entry_id, entry in entries.items():
                entry["entity_type"] = entity_type
                entry["id"] = entry_id
                ofd.write(json.dumps(entry) + "\n")
