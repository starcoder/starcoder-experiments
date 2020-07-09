import argparse
import gzip
import sys
import logging
import numpy
import sklearn.cluster
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-s", "--spec", dest="spec", help="Input file")
    parser.add_argument("-r", "--reduction", dest="reduction", type=float, default=0.9, help="")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with gzip.open(args.input, "rt") as ifd:        
        entities = [json.loads(line) for line in ifd]
    bottlenecks = numpy.array([e["bottleneck"] for e in entities])
    logging.info("Bottlenecks: {}".format(bottlenecks.shape))

    clusters = {}
    clusterer = sklearn.cluster.AgglomerativeClustering(n_clusters=round(args.reduction * bottlenecks.shape[0]),
                                                        linkage="single").fit(bottlenecks)
    #clusterer = sklearn.cluster.KMeans(n_clusters=round(args.reduction * bottlenecks.shape[0])).fit(bottlenecks)
    cids = clusterer.labels_
    for i, cid in enumerate(cids):
        clusters[cid] = clusters.get(cid, [])
        clusters[cid].append(entities[i])
    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps(list(clusters.values()), indent=2))
