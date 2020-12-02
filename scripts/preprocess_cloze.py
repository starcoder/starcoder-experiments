import gzip
import argparse
import zipfile
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    entities = {k : {} for k in ["story", "line", "character", "annotator", "annotation"]}
    with zipfile.ZipFile(args.inputs[0], "r") as izfd:
        with izfd.open("json_version/annotations.json", "r") as ifd:
            for story_index, (_story_id, story) in enumerate(json.loads(ifd.read()).items()):
                split = story["partition"]
                title = story["title"]
                story_id = "story {}".format(story_index)
                entities["story"][story_id] = {
                    "split" : split,
                    "title" : title
                }
                for line_num, line in story["lines"].items():
                    line_num = int(line_num)
                    line_id = "line {} {}".format(story_index, line_num)
                    text = line["text"]
                    entities["line"][line_id] = {
                        "text" : text,
                        "line_of" : story_id,
                    }
                    for char_name, char in line["characters"].items():
                        if char["app"] == True:
                            char_id = "char {} {}".format(story_index, char_name)                        
                            entities["character"][char_id] = entities["character"].get(
                                char_id, 
                                {
                                    "name" : char_name,
                                    "occurs_in" : story_id,
                                    "mentioned_in" : []
                                }
                            )
                            entities["character"][char_id]["mentioned_in"].append(line_id)
                            for ann_name, ann in char.get("emotion", {}).items():
                                ann_id = ann_name
                                entities["annotator"][ann_id] = {
                                    "username" : ann_id,
                                }
                                for stype, vals in [x for x in ann.items() if x[0] != "text"]:
                                    for val in vals:
                                        entities["annotation"][len(entities["annotation"])] = {
                                            "ann_type" : "motiv",
                                            "ann_subtype" : stype,
                                            "value" : val,
                                            "explanation" : " ".join(ann["text"]),
                                        }
                            for ann_name, ann in char.get("motiv", {}).items():
                                ann_id = ann_name
                                entities["annotator"][ann_id] = {
                                    "username" : ann_id,
                                }
                                for stype, vals in [x for x in ann.items() if x[0] != "text"]:
                                    for val in vals:
                                        entities["annotation"][len(entities["annotation"])] = {
                                            "ann_type" : "motiv",
                                            "ann_subtype" : stype,
                                            "value" : val,
                                            "explanation" : " ".join(ann["text"]),
                                        }

                    # emotion motiv
                    # plutchik
                    # maslow reiss text

    with gzip.open(args.output, "wt") as ofd:
        for entity_type, es in entities.items():
            for eid, entity in es.items():
                entity["id"] = eid
                entity["entity_type"] = entity_type
                ofd.write(json.dumps(entity) + "\n")
