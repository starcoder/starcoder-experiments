import argparse
import gzip
import json
import xml.etree.ElementTree as et
import re
import os.path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="inputs", help="Input files")    
    parser.add_argument("-s", "--schema", dest="schema", help="Schema file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()   

    entities = {k : {} for k in ["person", "paper", "volume", "collection", "publisher"]}
    
    for fname in args.inputs:
        with open(fname, "rt") as ifd:
            x = et.fromstring(ifd.read())
            coll_id = os.path.basename(fname)
            entities["collection"][coll_id] = entities["collection"].get(
                coll_id,
                {
                    "contains_volume" : []
                }
            )
            for vol in x.findall(".//volume[@id]"):
                vol_id = "{} {}".format(coll_id, vol.get("id"))
                entities["collection"][coll_id]["contains_volume"].append(vol_id)
                
                meta = vol.find("meta")
                venue = meta.find("booktitle")
                month = meta.find("month")
                year = meta.find("year")
                if year != None:
                    year = "".join(year.itertext()).strip()
                editor = meta.find("editor")
                publisher = meta.find("publisher")
                    
                address = meta.find("address")
                entities["volume"][vol_id] = entities["volume"].get(
                    vol_id,
                    {
                        "title" : [],
                        "address" : re.sub(r"\s+", " ", "".join(address.itertext())).strip() if address != None else None,
                        "date" : year,
                        "published_by" : [],
                        "contains_paper" : [],
                        "edited_by" : [],
                    }
                )
                if publisher != None and publisher.text != None:
                    publisher_name = publisher.text
                    publisher_id = publisher_name
                    entities["publisher"][publisher_id] = entities["publisher"].get(
                        publisher_id,
                        {
                            "name" : publisher_name,
                        }
                    )
                    entities["volume"][vol_id]["published_by"].append(publisher_id)
                for editor in vol.findall(".//editor"):
                    first_name = editor.get("first")
                    last_name = editor.get("last")
                    first_name = first_name.text if first_name != None else None
                    last_name = last_name.text if last_name != None else None
                    person_id = "person {} {}".format(first_name, last_name)
                    entities["person"][person_id] = entities["person"].get(
                        person_id,
                        {
                            "first_name" : first_name,
                            "last_name" : last_name,
                        }
                    )
                    entities["volume"][vol_id]["edited_by"].append(person_id)
                    
                
                for paper in vol.findall(".//paper[@id]"):
                    paper_id = "{} {}".format(vol_id, paper.get("id"))
                    title = paper.find(".//title")                    
                    pages = paper.find("pages")
                    if pages != None:
                        m = re.match(r".*?(\d+)\D+(\d+).*?", "".join(pages.itertext()))
                        try:
                            start = int(m.group(1))
                            end = int(m.group(2))
                            pages = end - start
                        except:
                            pages = None
                            pass
                    abstract = paper.find("abstract")
                    entities["paper"][paper_id] = entities["paper"].get(
                        paper_id,
                        {
                            "title" : re.sub(r"\s+", " ", "".join(title.itertext())).strip(),
                            "abstract" : re.sub(r"\s+", " ", "".join(paper.itertext() if paper else [])).strip(),
                            "length" : pages,
                            "written_by" : [],
                        }
                    )
                    for author in paper.findall(".//author"):

                        first_name = author.get("first")
                        last_name = author.get("last")
                        first_name = first_name.text if first_name != None else None
                        last_name = last_name.text if last_name != None else None
                        person_id = "person {} {}".format(first_name, last_name)
                        entities["person"][person_id] = entities["person"].get(
                            person_id,
                            {
                                "first_name" : first_name,
                                "last_name" : last_name,
                            }
                        )
                        entities["paper"][paper_id]["written_by"].append(person_id)

    with gzip.open(args.output, "wt") as ofd:
        for et, ets in entities.items():
            for eid, entity in ets.items():
                entity["id"] = eid
                entity["entity_type"] = et
                ofd.write(json.dumps({k : v for k, v in entity.items() if v != None}) + "\n")
