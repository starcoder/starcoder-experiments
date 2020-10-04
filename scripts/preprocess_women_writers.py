import argparse
import tarfile
import os.path
import re
import gzip
import json
import xml.etree.ElementTree as et


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    entities = {}
    ns = "{http://www.wwp.northeastern.edu/ns/textbase}"
    with tarfile.open(args.inputs[0], "r:gz") as ifd:
        for mem in ifd.getmembers():
            if mem.isfile() and mem.name != "women_writers/personography.xml":
                content = ifd.extractfile(mem)
                xml = et.parse(content)
                author = xml.find(".//{0}titleStmt/{0}author/{0}persName".format(ns)).text
                title = xml.find(".//{0}title[@type='main']".format(ns)).text
                #publisher
                year = xml.find(".//{0}imprint/{0}date")
                entities[author] = {"entity_type" : "author", "author_name" : author}
                entities[title] = {"entity_type" : "document", "document_title" : title} #, "written_by" : author}
                for text in xml.findall(".//{0}text".format(ns)):
                    divs = text.findall(".//{0}div".format(ns))
                    if len(divs) > 0:
                        section_id = "{}_{}".format(title, [v for k, v in text.attrib.items() if k.endswith("id")][0])
                        entities[section_id] = {"entity_type" : "section", "section_name" : section_id, "section_of" : title}
                        for i, div in enumerate(divs):
                            subsection_id = "{}_{}_{}".format(title, section_id, i)
                            text = re.sub(r"\s+", " ", " ".join(list(div.itertext()))).strip()
                            subsection_type = div.attrib.get("type")
                            entities[subsection_id] = {"entity_type" : "subsection",
                                                       "subsection_text" : text,
                                                       "subsection_of" : section_id,
                                                       "author_name" : author}
                            if subsection_type != None:
                                entities[subsection_id]["subsection_type"] = subsection_type

    with gzip.open(args.output, "wt") as ofd:
        for eid, entity in entities.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")
