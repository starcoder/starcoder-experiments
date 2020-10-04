import argparse
import gzip
import bz2
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="inputs", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    entities = {}
    with gzip.open(args.output, "wt") as ofd:
        for fname in args.inputs:
            with gzip.open(fname, "rt") as ifd:
                for i, line in enumerate(ifd):
                    j = json.loads(line)
                    author_id = (j["author_fullname"] if j["author_fullname"] != None else "_").split("_")[-1]
                    subreddit_id = j["subreddit_id"].split("_")[-1]
                    document_id = j["id"].split("_")[-1]
                    if author_id != "":
                        entities[author_id] = entities.get(author_id, {"id" : author_id,
                                                                       "entity_type" : "user",
                                                                       "user_name" : j["author"],
                        })
                    entities[subreddit_id] = entities.get(subreddit_id, {"id" : subreddit_id,
                                                                         "entity_type" : "subreddit",
                                                                         "subreddit_name" : j["subreddit"]})

                    document = {
                        "id" : document_id.split("_")[-1],
                        "entity_type" : "submission" if "selftext" in j else "comment",
                        "response_to" : j.get("parent_id", "").split("_")[-1] if j.get("parent_id", 1) != j.get("link_id", 2) else "",
                        "for_submission" : j.get("link_id", "").split("_")[-1] if "link_id" in j else "",
                        "posted_in" : subreddit_id,
                        "score" : j["score"],
                        "gilded" : j["gilded"],
                        #"edited" : str(j["edited"]),
                        "text" : j["body"] if "body" in j else j["selftext"],
                        "title" : j["title"] if "title" in j else "",
                        "written_by" : author_id if "body" in j else "",
                        "submitted_by" : author_id if "selftext" in j else "",
                        "creation_time" : j["created_utc"],
                    }

                    document = {k : v for k, v in document.items() if v not in ["", None]}
                    
                    ofd.write(json.dumps(document) + "\n")

        for eid, entity in entities.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")
    
    
