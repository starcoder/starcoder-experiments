import argparse
import json
import gzip
import numpy
from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", help="Input file")
    parser.add_argument("--id_file", dest="id_file", help="Entity IDs")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    if args.id_file != None:
        with gzip.open(args.id_file, "rt") as ifd:
            ids = json.loads(ifd.read())

    with gzip.open(args.input, "rt") as ifd:
        pixels = []
        for line in ifd:
            j = json.loads(line)
            if args.id_file == None or j["original"]["id"] in ids:
                if "image" in j:
                    r = numpy.array(j["image"], dtype=numpy.uint8)
                    pixels.append(r)
                elif "image" in j.get("original", []):
                    r = numpy.array(j["reconstruction"]["image"], dtype=numpy.uint8)
                    o = numpy.array(j["original"]["image"], dtype=numpy.uint8)
                    pixels.append(numpy.concatenate((o, r), axis=1))
        pixels = numpy.concatenate(pixels, axis=0)
        im = Image.fromarray(pixels, mode="RGB")
        im.save(args.output)