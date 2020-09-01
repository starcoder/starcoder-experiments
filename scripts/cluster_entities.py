import argparse
import gzip
import sys
import logging
import numpy
import sklearn.cluster
import json
from starcoder.schema import Schema
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-s", "--schema", dest="schema", help="Input file")
    parser.add_argument("-r", "--reduction", dest="reduction", type=float, default=0.9, help="")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.schema, "rt") as ifd:
        schema = Schema(json.loads(ifd.read()))
    
    def run_clustering(single_type_entities):
        bottlenecks = numpy.array([e["bottleneck"] for e in single_type_entities])
        logging.info("Bottlenecks: {}".format(bottlenecks.shape))
        if len(single_type_entities) == 1:
            return [single_type_entities]
        else:
            clusters = {}
            clusterer = sklearn.cluster.AgglomerativeClustering(n_clusters=round(args.reduction * bottlenecks.shape[0]),
                                                                linkage="single").fit(bottlenecks)
            #clusterer = sklearn.cluster.KMeans(n_clusters=round(args.reduction * bottlenecks.shape[0])).fit(bottlenecks)
            cids = clusterer.labels_
            for i, cid in enumerate(cids):
                clusters[cid] = clusters.get(cid, [])
                clusters[cid].append(single_type_entities[i])
            return list(clusters.values())

    entities = {}
    with gzip.open(args.input, "rt") as ifd:        
        for line in ifd:
            entity = json.loads(line)
            entity_type_name = entity["original"][schema.entity_type_field.name]
            entities[entity_type_name] = entities.get(entity_type_name, []) + [entity]

    clusters = sum([run_clustering(v) for v in entities.values()], [])

    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps(clusters, indent=2))
