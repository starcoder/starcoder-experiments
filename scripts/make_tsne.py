import argparse
from sklearn.manifold import TSNE
import gzip
import json
import numpy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", dest="schema", help="Input file")
    parser.add_argument("--data", dest="data", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    args = parser.parse_args()

    with open(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())
    eidf = schema["meta"]["id_field"]
    etf = schema["meta"]["entity_type_field"]
    
    data = {}
    indices = {}
    with gzip.open(args.data, "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            orig = j["original"]
            eid = orig[eidf]
            et = orig[etf]
            data[eid] = j
            indices[et] = indices.get(et, [])
            indices[et].append(eid)

    with gzip.open(args.output, "wt") as ofd:
        for et, eids in indices.items():
            print(et)
            if len(eids) == 1:
                continue
            bns = numpy.asarray([data[eid]["bottleneck"] for eid in eids])
            emb = TSNE().fit_transform(bns)
            for eid, row in zip(eids, emb):
                data[eid]["tsne"] = row.tolist()
                ofd.write(json.dumps(data[eid]) + "\n")

