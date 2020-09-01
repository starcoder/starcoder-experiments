import argparse
import gzip
import json
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("-s", "--schema", dest="schema", help="Schema file")
    args = parser.parse_args()

    with gzip.open(args.schema, "rb") as ifd:
        schema = pickle.load(ifd)
    
    with gzip.open(args.input, "rb") as ifd:
        clusters = json.load(ifd)
            
    with open(args.output, "wt") as ofd:
        for i, cluster in enumerate(clusters):            
            members = set([tuple(sorted([(k, v) for k, v in m["original"].items() if k not in [schema.id_field.name] + list(schema.relationship_fields.keys())])) for m in cluster])
            if len(members) == 1:
                continue

            ofd.write("Cluster #{}\n".format(i))
            for member in members:
                entity = {k : v for k, v in member}
                ofd.write("\t{}\n".format({k : v for k, v in entity.items()}))
