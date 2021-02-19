import argparse
import logging
import os.path
import json
import gzip
from datetime import date
from geocoding import Geocoder


relations = {
    ("voyage", "source") : "voyage_recorded_in_entry",
    ("notice", "source") : "notice_recorded_in_entry",
    ("voyage", "vessel") : "used_vessel",
    ("voyage", "shipper") : "shipped_by",
    ("slave", "consignor") : "consigned_by",
    ("slave", "owner") : "owned_by",
    ("voyage", "captain") : "captained_by",
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
    parser.add_argument("--location_cache", dest="location_cache", help="")
    args, rest = parser.parse_known_args()

    geocoder = Geocoder(args.location_cache)
        
    logging.basicConfig(level=logging.INFO)

    all_entities = []
    with gzip.open(args.inputs[0], "rt") as ifd:
        for i, line in list(enumerate(ifd)):
            if i > 10000:
                break
            entities = {}
            for k, v in json.loads(line).items():
                entity_type = k.split("_")[0]
                entities[entity_type] = entities.get(entity_type, {})
                if k in ["notice_event_date", "notice_date", "voyage_manifest_date", "voyage_arrival_date", "voyage_departure_date"]:
                    v = date.fromordinal(v).strftime("%d-%b-%Y")
                entities[entity_type][k] = v
                if k.endswith("location") or k == "jail_state":
                    g = geocoder(v)
                    if g != None:
                        entities[entity_type]["{}_coordinates".format(k)] = g
            for k in list(entities.keys()):
                entities[k]["id"] = "{}_{}".format(k, i)
                entities[k]["entity_type"] = k
            for (s, t), n in relations.items():
                if s in entities and t in entities:
                    entities[s][n] = entities[t]["id"]
            for v in entities.values():
                for name in ["voyage_manifest_count", "voyage_duration"]:
                    if name in v:
                        v[name] = float(v[name])
                all_entities.append(v)

    with gzip.open(args.output, "wt") as ofd:
        for e in all_entities:
            if "source_row" in e:
                e["source_row"] = str(e["source_row"])
            ofd.write(json.dumps(e) + "\n")
