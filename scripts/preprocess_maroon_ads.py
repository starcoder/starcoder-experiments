import json
import argparse
import csv
import sys
import gzip

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)

    data = []
    for fname in args.inputs:
        with open(fname, "rt") as ifd:
            for r, row in enumerate(csv.DictReader(ifd, delimiter="\t")):
                entities = {t : {"entity_type" : t, "id" : "row_{}_{}".format(r, t)} for t in ["notice", "city", "slave", "owner"]}
                for k, v in mapping.items():
                    entity_type = v.split("_")[0]
                    if row[k] != "":
                        try:
                            entities[entity_type][v] = float(row[k]) if schema[v]["type"] == "numeric" else row[k]
                        except:
                            print("Couldn't parse value '{}' for field '{}'", row[k], v)
                if row["Livre Equivalent"] != "":
                    entities["notice"]["notice_reward"] = float(row["Livre Equivalent"])
                elif row["Currency Specified Reward"].lower() == "livre":
                    entities["notice"]["notice_reward"] = float(row["Specified Reward"])
                entities["owner"]["submitted"] = entities["notice"]["id"]
                entities["slave"]["reported_in"] = entities["notice"]["id"]
                entities["city"]["location_of"] = entities["notice"]["id"]
                data += [v for v in entities.values()]
        
    with gzip.open(args.output, "wt") as ofd:
        for entity in data:
            ofd.write(json.dumps(entity) + "\n")
