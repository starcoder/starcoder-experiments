import argparse
import csv
import gzip
import json
import re
from geocoding import Geocoder



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--location_cache", dest="location_cache", help="")
    args, rest = parser.parse_known_args()

    geocoder = Geocoder(args.location_cache)

    entities = {"person" : {}, "contract" : {}, "location" : {}}
    with gzip.open(args.inputs[0], "rt") as ifd:
        for i, row in enumerate(csv.DictReader(ifd)):
            consentor, assignor, bound, bound_to, contract = {}, {}, {}, {}, {}
            date = "{Day}-{Month}-{Year}".format(**row) if all([row[k] for k in ["Year", "Month", "Day"]]) else None
            dep_loc_str = row["Departure Location (\"From the Port of\")"]
            bond_loc_str = row["Location of Bondage"]

            departure_location = {"address" : "{}".format(dep_loc_str)} if dep_loc_str else {}
            bondage_location = {"address" : "{}".format(bond_loc_str)} if bond_loc_str else {}
            assignor = {"name" : row["Assigned By"]} if row["Assigned By"] else {}
            bound = {"name" : "{} {}".format(row["First & Middle Name"], row["Last Name"])}
            bound_to = {"name" : row["Bound To"]} if row["Bound To"] else {}
            m = re.match(r"([a-z\s]+\s+)?([A-Z].*)", row["With Consent of"])
            if m:
                consentor = {"name" : m.group(2)}
                cr = m.group(1)
            else:
                consentor = {"name" : row["With Consent of"]} if row["With Consent of"] else {}
                cr = None

            contract = {
                "date" : date,
                "binds_as" : row["Bound As"],
                "binds" : bound.get("name"),
                "binds_to" : bound_to.get("name"),
                "with_consent_of" : consentor.get("name"),
                "assigned_by" : assignor.get("name"),
                "departed_from" : "loc {}".format(departure_location.get("address")) if departure_location != {} else None,
                "located_at" : "loc {}".format(bondage_location.get("address")) if bondage_location != {} else None,
                "at_expiration" : row["At Expiration"],
                "to_be_found" : row["To Be Found"],
                "to_be_taught" : row["To Be Taught"],
                "term" : row["Term"],
                "start" : row["Start Date of Contract"],
                "consentor_role" : cr
            }
            m = re.match(r"^(.*?)([\d\.]+)(.*?)$", row["Amount"])

            if m:
                nums = [int(x) for x in m.group(2).split(".") if x != ""]
                curr = m.group(1) + m.group(3)
                if "£" in curr or "pound" in curr or curr.strip() == "":
                    l, s, p = nums + [0 for i in range(3 - len(nums))]
                    contract["amount"] = l * 240 + s * 12 + p
                elif "s" in curr or "/" in curr:
                    s, p = nums + [0 for i in range(2 - len(nums))]
                    contract["amount"] = s * 12 + p
                elif "pence" in curr:
                    contract["amount"] = nums[0]

            entities["contract"]["contract {}".format(i)] = contract
            for loc in [departure_location, bondage_location]:
                if loc:
                    g = geocoder(loc["address"])
                    if g != None:
                        loc["coordinates"] = g
                    entities["location"]["loc {}".format(loc["address"])] = loc
            for person in [bound, bound_to, assignor, consentor, ]:
                if person:
                    entities["person"][person["name"]] = person

    with gzip.open(args.output, "wt") as ofd:
        for et, eot in entities.items():
            for eid, e in eot.items():
                e["id"] = eid
                e["entity_type"] = et
                ofd.write(json.dumps({k : v for k, v in e.items() if v not in [None]}) + "\n")
