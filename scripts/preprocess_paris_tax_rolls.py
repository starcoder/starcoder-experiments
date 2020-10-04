import argparse
import tarfile
import os.path
import re
import gzip
import json
import xml.etree.ElementTree as et
from geocoding import Geocoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--location_cache", dest="location_cache", help="")
    args, rest = parser.parse_known_args()

    geocoder = Geocoder(args.location_cache)

    # div-parish div-street list-payers
    # persName roleName payment status
    ns = "{http://www.tei-c.org/ns/1.0}"
    entities = {}
    debt_counter = 0
    with tarfile.open(args.inputs[0], "r:gz") as ifd:
        for mem in ifd.getmembers():
            if mem.isfile() and re.match(r".*taxroll_\d+\.xml$", mem.name):
                year = os.path.splitext(mem.name)[0].split("_")[-1]
                content = ifd.extractfile(mem)
                roll_id = "year_{}".format(year)
                entities[roll_id] = {"entity_type" : "tax_roll", "year" : year}
                xml = et.parse(content)
                for parish in xml.getroot().findall(".//{}div[@type='parish']".format(ns)):
                    parish_id = parish.find("{0}div/{0}head/{0}rs".format(ns)).attrib["ref"]
                    entities[parish_id] = {"entity_type" : "parish", "parish_name" : parish_id}
                    for street in parish.findall(".//{}div[@type='street']".format(ns)):
                        street_id = street.find("{0}head/{0}rs".format(ns)).attrib["ref"].split(" ")[0]
                        street_name = " ".join(street_id.split("_")[1:])
                        entities[street_id] = {"entity_type" : "street", "street_name" : street_name}
                        g = None
                        while g == None:
                            try:
                                g = geocoder("{}, Paris, France".format(street_name))
                            except:
                                g = None
                        entities[street_id]["street_coordinates"] = g #{"latitude" : g.latitude, "longitude" : g.longitude}
                        for payers in street.findall(".//{}list[@type='payers']".format(ns)):
                            for item in payers.findall(".//{}item".format(ns)):
                                payer = item.find(".//{0}seg[@type='entry']/{0}persName".format(ns))
                                amount = item.find(".//{0}seg[@type='payment']/{0}measure".format(ns))
                                status = item.find(".//{0}seg[@type='status']".format(ns))
                                role = item.find(".//{0}roleName".format(ns))
                                if payer != None:
                                    name = re.sub(r"\s+", " ", "".join(payer.itertext())).strip()
                                else:
                                    continue
                                if amount != None:
                                    unit = amount.attrib["unit"]
                                    amount = float(amount.attrib["quantity"])
                                if status != None:
                                    status = "".join(status.itertext()).strip()
                                role_id = None
                                if role != None and "role" in role.attrib:
                                    role_type = role.attrib["type"]
                                    role_name = role.attrib["role"]
                                    role_id = "{}_{}".format(role_type, role_name)
                                debt_counter += 1
                                debt_id = "debt_{}".format(debt_counter)                                
                                person_id = "person_{}_{}".format(name, street_id)
                                entities[person_id] = {"entity_type" : "person", "person_name" : name}
                                entities[debt_id] = {"entity_type" : "debt", "currency" : unit, "status" : status}
                                if amount != None:
                                    entities[debt_id]["amount"] = amount
                                entities[debt_id]["recorded_in"] = roll_id
                                entities[debt_id]["owed_by"] = person_id
                                entities[street_id]["part_of"] = parish_id
                                entities[person_id]["lives_on"] = street_id
                                if role_id != None:
                                    entities[role_id] = {"entity_type" : "role", "role_name" : role_name}
                                    entities[role_id]["role_type"] = role_type
                                    entities[person_id]["acts_as"] = role_id
                                
    with gzip.open(args.output, "wt") as ofd:
        for eid, entity in entities.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")
    
