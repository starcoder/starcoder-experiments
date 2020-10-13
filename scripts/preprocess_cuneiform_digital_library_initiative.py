import argparse
import gzip
import csv
import re
import json
from geocoding import Geocoder

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    parser.add_argument("--location_cache", dest="location_cache", help="")    
    args, rest = parser.parse_known_args()

    geocoder = Geocoder(args.location_cache)
    
    fields = {}
    with gzip.open(args.inputs[0], "rt") as ifd:        
        for row in csv.DictReader(ifd):
            eid = str(int(row["id"]))
            assert eid not in fields
            fields[eid] = row

    entities = {"inscription" : {}}
    with gzip.open(args.inputs[1], "rt") as ifd:
        for span in re.split(r"^\&P", ifd.read().lstrip("&P"), flags=re.M|re.S):
            m = re.match(r"""
\s*(?P<cid>\S+)\s*(=\s*(?P<oid>.*?))?\n
(#atf: lang (?P<lang>\S+)\n
\@(?P<type>.*?)\n)?
(?P<rest>.*)
""", span, flags=re.X|re.S)            
            eid = str(int(m.group("cid")))
            if eid in fields:
                entity = {"text" : [], "translation" : []}
                for n in re.finditer(r"^(\d+.*?)\n(#tr.en*?\n)?", m.group("rest"), re.M):
                    if n.group(2) != None:
                        entity["translation"].append(n.group(2))
                    entity["text"].append(n.group(1))
                    #for field in ["date_of_origin"]: #, "dates_referenced"]:
                    #    pass
                    for field in ["genre", "language", "material", "object_type", "period", "provenience", "subgenre"]: # seal_id
                        entity[field] = fields[eid].get(field, "")
                        pass
                    for field in ["thickness", "width", "height"]:
                        try:
                            entity[field] = float(fields[eid].get(field, ""))
                        except:
                            pass
                    o = re.match(r".*\(mod. (.*)\).*", fields[eid].get("provenience", ""))
                    if o:
                        g = geocoder(o.group(1))
                        if g:
                            entity["location"] = g
                entity["text"] = "".join(entity["text"])
                entity["translation"] = "".join(entity["translation"])                
                entities["inscription"][eid] = {k : v for k, v in entity.items() if v}
    with gzip.open(args.output, "wt") as ofd:
        for entity_type, es in entities.items():
            for eid, entity in es.items():
                entity["id"] = eid
                entity["entity_type"] = entity_type
                ofd.write(json.dumps(entity) + "\n")




