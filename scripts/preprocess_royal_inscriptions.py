import argparse
import tarfile
import os.path
import re
import gzip
import json

# RIME 1.09.05.01, ex. 01 --> E=early, 1=v1, 9=dynasty, 5=ruler, 1=text, *=exemplar 1
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    entities = {}
    with tarfile.open(args.inputs[0], "r:gz") as ifd:
        for mem in ifd.getmembers():
            if mem.isfile():
                rim = os.path.splitext(os.path.basename(mem.name))[0]
                period, volume, _, dynasty, ruler, text_num, wit = re.match(r"^(.)(\d+)(:|\.)(\d+)\.(\d+)\.(\d+)(.*?)?$", rim).groups()
                period_id = "period_{}".format(period)
                dynasty_id = "dynasty_{}_{}".format(period, dynasty)
                ruler_id = "ruler_{}_{}_{}".format(period, dynasty, ruler)
                text_id = "text_{}_{}_{}_{}".format(period, dynasty, ruler, text_num)
                witness_id = "witness_{}_{}_{}_{}_{}".format(period, dynasty, ruler, text_num, wit)
                content = ifd.extractfile(mem).read().decode("utf-8")
                entities[period_id] = {"entity_type" : "period"}
                entities[dynasty_id] = {"entity_type" : "dynasty", "from_period" : period_id}
                entities[ruler_id] = {"entity_type" : "ruler", "from_dynasty" : dynasty_id}
                entities[text_id] = {"entity_type" : "text", "for_ruler" : ruler_id}
                entities[witness_id] = {"entity_type" : "witness", "witness_of" : text_id, "content" : content}

    with gzip.open(args.output, "wt") as ofd:
        for eid, entity in entities.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")
