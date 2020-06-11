import argparse
import logging
import os.path
import json
import gzip

relations = {
    ("voyage", "source") : "voyage_recorded_in_entry",
    ("notice", "source") : "notice_recorded_in_entry",
    ("voyage", "vessel") : "used_vessel",
    ("voyage", "shipper") : "shipped_by",
    ("slave", "consignor") : "consigned_by",
    ("slave", "owner") : "owned_by",
    ("vessel", "captain") : "captained_by",
    ("slave", "voyage") : "passenger_on",
    ("notice", "slave") : "reports_slave",
    ("notice", "gazette") : "published_in",
    ("slave", "jail") : "imprisoned_in",
    ("notice", "author") : "written_by",
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()    
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument(dest="inputs", nargs="+")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    all_entities = []
    with gzip.open(args.inputs[0], "rt") as ifd:
        for i, line in enumerate(ifd):
            entities = {}
            for k, v in json.loads(line).items():
                entity_type = k.split("_")[0]
                entities[entity_type] = entities.get(entity_type, {})
                entities[entity_type][k] = v
            for k in list(entities.keys()):
                entities[k]["id"] = "{}_{}".format(k, i)
                entities[k]["entity_type"] = k
            for (s, t), n in relations.items():
                if s in entities and t in entities:
                    entities[s][n] = entities[t]["id"]
            for v in entities.values():
                all_entities.append(v)

    with gzip.open(args.output, "wt") as ofd:
        for e in all_entities:
            ofd.write(json.dumps(e) + "\n")
