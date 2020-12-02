import argparse
import gzip
from xml.etree import ElementTree as et
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--location_cache", dest="location_cache", help="")
    args, rest = parser.parse_known_args()

    entities = {
        "document" : {},
        "sura" : {},
        "aya" : {},
    }
    with gzip.open(args.inputs[0], "rt") as ifd:
        xml = et.parse(ifd)
        document_id = "quran 1"
        entities["document"][document_id] = {
            "title" : "quran"
        }
        for sura in xml.findall("sura"):
            sura_id = "sura {}".format(sura.get("index"))
            entities["sura"][sura_id] = {
                "number" : int(sura.get("index")),
                "name" : sura.get("name"),
                "sura_of" : document_id,
            }
            for aya in sura.findall("aya"):
                aya_id = "aya {}".format(aya.get("index"))
                entities["aya"][aya_id] = {
                    "number" : int(aya.get("index")),
                    "text" : aya.get("text"),
                    "aya_of" : sura_id,
                }
    with gzip.open(args.output, "wt") as ofd:
        for entity_type, es in entities.items():
            for eid, entity in es.items():
                entity["entity_type"] = entity_type
                entity["id"] = eid
                ofd.write(json.dumps(entity) + "\n")
