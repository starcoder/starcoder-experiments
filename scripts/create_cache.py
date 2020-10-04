import gzip
import argparse
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    items = []
    for fname in args.inputs:
        with gzip.open(fname, "rt") as ifd:
            for line in ifd:
                j = json.loads(line)
                for k, v in j.items():
                    if isinstance(v, dict) and "longitude" in v:
                        if "text" not in v:
                            print(fname)
                        items.append(v)
                        #if k.endswith("_coordinates"):#isinstance(v, dict) and "latitude" in v:
                        #f = "_".join(k.split("_")[:-1])
                        #if any([x in fname for x in ["affiches", "maroon", "entertaining", "paris"]]):                            
                        #    f += "_name"
                        #v["text"] = j[f]
                        #items.append(v)
    
    with gzip.open(args.output, "wt") as ofd:
        for item in items:
            ofd.write("{latitude}\t{longitude}\t{text}\n".format(**item))
