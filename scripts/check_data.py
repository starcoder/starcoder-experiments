import argparse
import gzip
import json
import logging
import scipy.sparse
import numpy

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data", help="Input file")
    parser.add_argument("--schema", dest="schema", help="Input file")
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    logger.info("Examining schema '%s'", args.schema)

    with open(args.schema, "rt") as ifd:
        schema = json.load(ifd)

    entity_types = []
    for etn, etf in schema["entity_types"].items():
        entity_types.append(etn)

    
    adj = numpy.zeros(shape=(len(entity_types), len(entity_types)))
    for rfn, rf in schema["relationship_fields"].items():
        src = entity_types.index(rf["source_entity_type"])
        tgt = entity_types.index(rf["target_entity_type"])
        adj[src, tgt] += 1
        adj[tgt, src] += 1
    gr = scipy.sparse.csgraph.csgraph_from_dense(adj)
    ncomp, assignments = scipy.sparse.csgraph.connected_components(gr)
    if ncomp != 1:
        print(entity_types)
        print(assignments)
        raise Exception("Entity-types are not fully connected by relationships")

    idf = schema["meta"]["id_field"]
    etf = schema["meta"]["entity_type_field"]
    reported = set()
    seen = set()

    if args.data != None:
        with gzip.open(args.data, "rt") as ifd:
            for line in ifd:
                j = json.loads(line)
                et = j[etf]
                for k in [x for x in j.keys() if x not in [idf, etf]]:
                    seen.add(k)                    
                    if k not in schema["relationship_fields"] and (k not in schema["entity_types"][et]["data_fields"] or k not in schema["data_fields"]):
                        if (et, k) not in reported:
                            print("Undefined: {} {} for {}".format(args.data, k, et))
                            reported.add((et, k))
                        

        for k in schema["data_fields"].keys():
            if k not in seen:
                print("Unseen field: {} {}".format(args.data, k))
