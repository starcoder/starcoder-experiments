import numpy
import re
import argparse
import gzip
import json
import os.path
from PIL import Image

def fix_date(d):
    if d == None:
        return None
    else:
        m, d, y = map(int, j["date"].split("/"))
        return "{:02}/{:02}/{}".format(m, d, y)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    entities = {e : {} for e in ["crime", "person"]}
    with gzip.open(args.inputs[0], "rt") as ifd:
        for line in ifd:
            j = json.loads(line)
            ex_id = j["execution_id"]
            m, d, y = map(int, j["date"].split("/"))
            dem = j.get("Race and Gender of Victim", "").lower().strip()
            vgen = "female" if "female" in dem else "male" if "male" in dem else None
            vrace = dem if vgen == None else dem.replace(vgen, "").strip()
            crime = {
                "execution_date" : fix_date(j.get("date", None)), #"{:02}/{:02}/{}".format(m, d, y),
                "imprisoned_date" : fix_date(j.get("Date Received", None)),
                "offense_date" : fix_date(j.get("Date of Offense", None)),
                "summary" : j.get("Summary of Incident", None),
                "statement" : j.get("statement", None),
                "victim_race" : vrace,
                "county" : j.get("County", None),
                "victim_gender" : vgen,
            }
            entities["crime"][ex_id] = crime
            if "Height (in Feet and Inches)" in j:
                try:
                    ft, ins = re.split(r"\s+", re.sub(r"\D+", " ", j["Height (in Feet and Inches)"]).strip())
                    hgt = 12 * int(ft) + int(ins)
                except:
                    hgt = None
            else:
                hgt = None
            pris = {
                "height" : hgt,
                "weight" : int(re.sub(r"\D", "", j.get("Weight (in Pounds)", "0"))),
                "prior_occupation" : j.get("Prior Occupation", None),
                "prior_prison_record" : j.get("Prior Prison Record", None),
                "name" : re.sub(r"\s+", " ", "{} {}".format(j["first_name"], j["last_name"])),
                "age" : int(j["age"]),
                "race" : j.get("Race", None),
                "gender" : j.get("Gender", None),
                "hair_color" : j.get("Hair Color", None),
                "eye_color" : j.get("Eye Color", None),
                "native_county" : j.get("Native County", None),
                "native_state" : j.get("Native State", None),
                #"image" : j.get("suspect_image", None),
                "birth_date" : fix_date(j.get("Date of Birth", None)),
                "education" : j.get("Education Level (Highest Grade Completed)", None),
                "age_at_incarceration" : int(j.get("Age (when Received)", 0)),
                "age_at_offense" : int(j.get("Age (at the time of Offense)", 0)),
                "age_at_death" : int(j.get("age", 0)),
                "executed_for" : ex_id
            }
            if j.get("suspect_image", None) != None:
                image_fname = os.path.join(os.path.dirname(args.inputs[0]), "images", j["suspect_image"])
                if os.path.exists(image_fname):
                    im = Image.open(image_fname)
                    im.thumbnail((32, 32))
                    n = numpy.asarray(im)
                    m = numpy.zeros((32, 32, 3))
                    if len(n.shape) == 3:
                        m[0:n.shape[0], 0:n.shape[1], 0:] = n
                        pris["image"] = m.tolist()
                    #pris["image"] = n.tolist()
            pris["participated_in"] = entities["person"].get(pris["name"], {}).get("participated_in", [])
            entities["person"][pris["name"]] = pris
            for codef in j.get("Co-Defendants", "").split(" and "):
                if codef != "" and codef != "None":
                    cid = re.sub(r"\s+", " ", codef)
                    if cid not in entities["person"]:
                        entities["person"][cid] = {
                            "name" : cid
                        }
                    entities["person"][cid]["participated_in"] = entities["person"].get(cid, {}).get("participated_in", []) + [ex_id]

    with gzip.open(args.output, "wt") as ofd:
        for et, eot in entities.items():
            for eid, e in eot.items():
                #if e.get("image", None) == None:
                #    continue
                e = {k : v for k, v in e.items() if v}
                e["id"] = eid
                e["entity_type"] = et
                ofd.write(json.dumps(e) + "\n")

