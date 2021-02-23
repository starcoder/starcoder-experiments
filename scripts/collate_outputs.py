import re
import math
import argparse
import gzip
import logging
import json
import csv
import torch
from sklearn.metrics import f1_score
from jsonpath_ng.ext import parse

logger = logging.getLogger("collate_outputs")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-t", "--test", dest="test", default=None, help="Test IDs")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], help="Logging level")
    args, rest = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )
    
    if args.test != None:
        test_ids = set()    
        with gzip.open(args.test, "rt") as ifd:
            test_ids = json.loads(ifd.read())

    cols = set()
    data = []
    seen = {}
    pattern = parse("$..*")
    for i in range(len(args.inputs) // 2):
        sname = args.inputs[i * 2]
        oname = args.inputs[i * 2 + 1]
        conf = {}
        with open(sname, "rt") as ifd:
            schema = json.loads(ifd.read())
        for match in pattern.find(schema):
            if str(match.path) == "meta":
                full = str(match.full_path)
                for k, v in match.value.items():
                    key = "{}.{}".format(full, k)
                    conf[key] = v
                    seen[key] = seen.get(key, set())
                    seen[key].add(str(v))
        datum = {}
        with gzip.open(oname, "rt") as ifd:
            for line in ifd:
                entity = json.loads(line)
                if args.test == None or entity["original"]["id"] in test_ids:
                    for k, v in entity["original"].items():
                        tp = schema["properties"].get(k, {}).get("type")
                        if tp in ["scalar", "categorical", "distribution", "boolean"]:
                            rv = entity["reconstruction"][k] #.get(k, None)
                            datum[k] = datum.get(k, [])
                            datum[k].append((v, rv))

        for k in list(datum.keys()):
            cols.add(k)
            tp = schema["properties"].get(k, {}).get("type")
            if tp in ["categorical", "boolean"]:
                logger.info("Computing F-score for %d pairs of property '%s'",
                            len(datum[k]),
                            k
                        )
                datum[k] = f1_score(
                    [str(x) for x, _ in datum[k]],
                    [str(x) for _, x in datum[k]],
                    average="macro",
                )

            elif tp == "scalar":
                datum[k] = sum([math.sqrt((a - b)**2) for a, b in datum[k]]) / len(datum[k])
            elif tp == "distribution":
                poss = set()
                for (a, b) in datum[k]:
                    for j in list(a.keys()) + list(b.keys()):
                        poss.add(j)
                poss = list(poss)
                gold = torch.zeros(size=(len(datum[k]), len(poss)))
                guess = torch.zeros(size=(len(datum[k]), len(poss)))
                for j, (a, b) in enumerate(datum[k]):
                    gold[j] = torch.tensor([a.get(x, 0.0) for x in poss]) #x[1] for x in sorted(a.items())])
                    guess[j] = torch.tensor([b.get(x, 0.0) for x in poss]) #x[1] for x in sorted(a.items())])
                    #guess[i] = torch.tensor([x[1] for x in sorted(b.items())])
                datum[k] = torch.nn.functional.kl_div(guess, gold, reduction="batchmean").tolist()

        conf.update(datum) #.update(conf)        
        data.append(conf)
    #print([v for v in data if v["meta.depth"] == 0])
    cols = [k for k, v in seen.items() if len(v) > 1] + list(cols)
    with open(args.output, "wt") as ofd:
        cofd = csv.DictWriter(ofd, fieldnames=list(cols), delimiter="\t")
        cofd.writeheader()
        for row in data:
            cofd.writerow({k : v for k, v in row.items() if k in cols})
