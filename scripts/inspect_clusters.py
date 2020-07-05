import argparse
import gzip
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    rels = ["unary_arg", "left_arg", "right_arg"]
    
    with gzip.open(args.input, "rb") as ifd:
        clusters = json.load(ifd)
    lookup = {}
    for cluster in clusters:
        for entity in cluster:
            lookup[entity["id"]] = (entity["name"], entity["value"])
            
    with open(args.output, "wt") as ofd:
        for i, cluster in enumerate(clusters):
            if len(cluster) == 1:
                continue
            ofd.write("Cluster #{}\n".format(i))
            for entity in cluster:
                ofd.write("\t{}\n".format({k : v for k, v in entity.items() if k not in rels + ["_bottleneck", "_reconstruction"]}))
                for rel in rels:
                    if entity.get(rel, None) in lookup: #!= None:                        
                        ofd.write("\t\t{}\n".format(lookup[entity[rel]]))
