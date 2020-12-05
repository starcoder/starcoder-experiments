import argparse
import gzip
import json
import logging
import numpy
from sklearn.manifold import TSNE

logger = logging.getLogger("make_tsne")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", dest="schema", help="Input file")
    parser.add_argument("--data", dest="data", help="Input file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--log_level", dest="log_level", default="ERROR", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], help="Logging level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )

    with open(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())
    eidf = schema["meta"]["id_property"]
    etf = schema["meta"]["entity_type_property"]
    
    data = {}
    indices = {}
    with gzip.open(args.data, "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            orig = j["original"]
            eid = orig[eidf]
            et = j["original"][etf]
            data[eid] = {"bottleneck" : j["bottleneck"]}
            indices[et] = indices.get(et, [])
            indices[et].append(eid)

    with gzip.open(args.output, "wt") as ofd:
        for et, eids in indices.items():
            if len(eids) == 1:
                continue
            bns = numpy.asarray([data[eid]["bottleneck"] for eid in eids])
            logger.info("Bottleneck shape for entity-type '%s': %s", et, bns.shape)
            emb = TSNE().fit_transform(bns)
            for eid, row in zip(eids, emb):
                data[eid]["tsne"] = row.tolist()
                ofd.write(json.dumps(data[eid]) + "\n")

