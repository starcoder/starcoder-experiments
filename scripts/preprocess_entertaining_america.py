import argparse
import logging
import os.path
import json
import gzip
import re
import datetime
import calendar
import csv

entity_types = {
    "performance" : ["amateur", "local_amateur", "audience_size", "price", "popular_performance_other", "play_name", "performance_notes", "performance_quality", "outdoor_type", "event_type_other", "event_type_minstrel", "event_type_classical", "event_type_brass", "ethnicity", "ethnicity_list", "ethnicity_other", "event_type", "ethnic_comedy", "ethnic_comedy_other", "classical_performance_other", "class", "song_title_female", "song_title_male", "event_type_vaudeville"],
    "performer" : ["performer_name"],
    "location" : ["venue_chain", "location_venue", "location_venue_other", "location_name", "location_exact", "location_owner"],
    "notice" : ["repeat", "repeat_notice", "source_type", "source_type_other", "notes"],
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+")
    parser.add_argument("-o", "--output", dest="output")    
    args, rest = parser.parse_known_args()

    logging.basicConfig(level=logging.INFO)

    lookup = {}
    entities = {}
    fields = {}
    notice_count = 0
    with gzip.open(args.inputs[0], "rt") as ifd:
        for row in csv.DictReader(ifd, delimiter="\t"):
            row = {k.lower() : v for k, v in row.items()}
            notice_count += 1
            performance, location, notice = [{"entity_type" : "performance", "id" : "performance_{}".format(notice_count)},
                                             {"entity_type" : "location"},
                                             {"entity_type" : "notice", "id" : "notice_{}".format(notice_count)}]
            for field_name, v in row.items():
                v = v.strip()
                if v != "":
                    if field_name in ["price", "audience_size"]:
                        val = float(v)
                    else:
                        val = v.strip()
                    if field_name in entity_types["performance"]:
                        performance[field_name] = val
                    elif field_name in entity_types["location"]:
                        location[field_name] = val
                    elif field_name in entity_types["notice"]:
                        notice[field_name] = val
            other, primary = {}, {}
            if row["other_performers"] != "":
                other = {"entity_type" : "performer",
                         "id" : row["other_performers"],
                         "performer_name" : row["other_performers"]}
                performance["has_other_performer"] = other["id"]
            if row["primary_performer"] != "":
                primary = {"entity_type" : "performer",
                           "id" : row["primary_performer"],
                           "performer_name" : row["primary_performer"]}
                performance["has_primary_performer"] = primary["id"]
            if "location_name" in location: # and "location_venue" in location and "location_owner" in location:
                location["id"] = "_".join([location.get(x, "") for x in ["location_name", "location_venue", "location_owner"]])
                key = (location.get("location_name", ""),) #, , location.get("location_owner", ""))
                lookup[key] = lookup.get(key, []) + [location["id"]]
                performance["occurred_at"] = location["id"]
            performance["advertised_by"] = notice["id"]
            #print(location.get("location_name", "unknown").lower())
            for entity in [performance, location, notice, primary, other]:
                if "id" in entity:
                    entities[entity["id"]] = entity
    #print(lookup)
    with gzip.open(args.inputs[1], "rt") as ifd:
        for feat in json.loads(ifd.read())["features"]:
            key = (feat["properties"]["Location_name"],) #feat["properties"]["Location_Owner"])
            g = {"longitude" : feat["geometry"]["coordinates"][0], "latitude" : feat["geometry"]["coordinates"][1], "text" : feat["properties"]["Location_name"]}
            # source_date "Wed, 15 May 1872 04:56:02 GMT"
            if key in lookup and g["longitude"] != 0:
                for i in lookup[key]:
                    entities[i]["location_coordinates"] = g
                    #print(key, g)
            #if g["longitude"] != 0:
            #    print(key, g)
            #"Location_name", "Location_Owner", "Location_venue"
            #pass
        
    with gzip.open(args.output, "wt") as ofd:
        for eid, entity in entities.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")

