import json
import argparse
import csv
import sys
import gzip
import logging

mapping = {"Date" : "notice_date",
           "City" : "city_name",
           "Name" : "slave_name",
           "Race" : "slave_race",
           "Race Category" : "slave_race_category",
           "Gender" : "slave_gender",
           "Age" : "slave_age",
           "d'environ age" : "slave_age_is_approximate",
           "height" : "slave_height",
           "d'environ size" : "slave_height_is_approximate",
           "Time Escaped" : "notice_time_escaped",
           "Inspecific Time" : "notice_time_escaped_is_approximate",
           "Group Size" : "notice_escape_group_size",
           "Profession" : "slave_profession",
           "Known Associates?" : "slave_has_known_associates",
           "Slaver" : "owner_name",
           "Slaver Profession" : "owner_profession",
}

numeric_fields = [
    "notice_escape_group_size",
    "notice_reward",
    "slave_age",
    "slave_height",
]

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    seen_cities = set()
    data = []
    for fname in args.inputs:
        with open(fname, "rt") as ifd:
            for r, row in enumerate(csv.DictReader(ifd, delimiter="\t")):
                entities = {t : {"entity_type" : t, "id" : "row_{}_{}".format(r, t)} for t in ["notice", "city", "slave", "owner"]}
                for k, v in mapping.items():
                    entity_type = v.split("_")[0]
                    if row[k] != "":
                        try:
                            entities[entity_type][v] = float(row[k]) if v in numeric_fields else row[k]
                        except Exception as e:                            
                            logging.warning("Couldn't parse value '%s' for field '%s' of entity type '%s'", row[k], v, entity_type)
                            #raise e
                if row["Livre Equivalent"] != "":
                    entities["notice"]["notice_reward"] = float(row["Livre Equivalent"])
                elif row["Currency Specified Reward"].lower() == "livre":
                    entities["notice"]["notice_reward"] = float(row["Specified Reward"])
                entities["owner"]["submitted"] = entities["notice"]["id"]
                entities["slave"]["reported_in"] = entities["notice"]["id"]
                if "city_name" in entities["city"]:
                    entities["city"]["id"] = entities["city"]["city_name"]
                    entities["notice"]["posted_in"] = entities["city"]["id"]
                    if entities["city"]["id"] in seen_cities:
                        del entities["city"]
                    else:                        
                        seen_cities.add(entities["city"]["id"])
                else:
                    del entities["city"]
                data += [v for v in entities.values()]

    with gzip.open(args.output, "wt") as ofd:
        for entity in data:
            ofd.write(json.dumps(entity) + "\n")
