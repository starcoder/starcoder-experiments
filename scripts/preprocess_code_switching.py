import argparse
import gzip
import json
import numpy

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="inputs", help="Input files")    
    parser.add_argument("-s", "--schema", dest="schema", help="Schema file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()   

    lengths = []
    languages = {}
    
    with gzip.open(args.inputs[0], "rt") as ifd:
        for line in ifd:
            _, lang, text = line.strip().split("\t")
            languages[lang] = languages.get(lang, [])
            words = text.split()
            languages[lang] += words
            lengths.append(len(words))
    language_names = {i : k for i, k in enumerate(languages.keys())}
    
    docs = {}
    while any([len(v) > 0 for v in languages.values()]):
        total_count = numpy.random.randint(10, max(lengths))
        per_language = {}
        item = {"text" : []}
        for i, p in enumerate(numpy.random.dirichlet([0.01] * len(languages))):
            n = int(total_count * p)
            lang = language_names[i]
            words = languages[lang][:n]
            languages[lang] = languages[lang][n:]
            per_language[lang] = sum([len(w) for w in words])
            item["text"] += words
        item["text"] = " ".join(item["text"])
        if len(item["text"]) == 0:
            continue
        item["language_proportions"] = {l : c / len(item["text"]) for l, c in per_language.items()}
        docs[str(len(docs))] = item

    with gzip.open(args.output, "wt") as ofd:
        for eid, entity in docs.items():
            entity["id"] = eid
            entity["entity_type"] = "document"
            ofd.write(json.dumps(entity) + "\n")
