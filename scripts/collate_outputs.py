import argparse
import gzip
import json
from sklearn.metrics import f1_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-t", "--test", dest="test", help="Test IDs")
    parser.add_argument("-s", "--schema", dest="schema", help="Schema")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    with open(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())

    test_ids = set()
    with gzip.open(args.test, "rt") as ifd:
        for line in ifd:
            test_ids.add(line[:-1])

    data = {}
    for i in range(len(args.inputs) // 2):
        cname = args.inputs[i * 2]
        oname = args.inputs[i * 2 + 1]
        #entities = []
        with gzip.open(cname, "rt") as ifd:
            config = json.loads(ifd.read())
        #datum = {"BATCH_SIZE", "SPLITTER_CLASS", "BATCHIFIER_CLASS", "DEPTH"}
        datum = {"DEPTH" : config["DEPTH"]}
        with gzip.open(oname, "rt") as ifd:
            for line in ifd:
                entity = json.loads(line)
                if entity["original"]["id"] in test_ids and entity["original"]["entity_type"] in ["node", "tweet"]:
                    for k, v in entity["original"].items():
                        rv = entity["reconstruction"].get(k, None)
                        if rv != None:
                            tp = schema["data_fields"].get(k, {}).get("type")
                            if tp in ["scalar", "categorical"]:
                                datum[k] = datum.get(k, [])
                                datum[k].append((v, rv))
                    #entities.append(entity)
        #print(len(entities))
        for k in list(datum.keys()):
            tp = schema["data_fields"].get(k, {}).get("type")
            if tp == "categorical":
                datum[k] = f1_score(
                    [x for x, _ in datum[k]],
                    [x for _, x in datum[k]],
                    average="macro",
                )
        print(datum)
    
