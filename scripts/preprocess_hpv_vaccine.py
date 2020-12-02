import argparse
import gzip
import bz2
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="inputs", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    counts = {}
    entities = {}
    with gzip.open(args.output, "wt") as ofd:
        for fname in args.inputs:
            with gzip.open(fname, "rt") as ifd:
                for i, line in enumerate(ifd):
                    j = json.loads(line)
                    et = "submission" if "selftext" in j else "comment"
                    counts[et] = counts.get(et, 0) + 1
                    author_id = "author {}".format(j["author"]) #j["author_fullname"] if j.get("author_fullname", None) != None else j["author"]) #.split("_")[-1]
                    subreddit_id = j["subreddit_id"] #.split("_")[-1]
                    document_id = j["id"] #.split("_")[-1]
                    parent_id = j["parent_id"] if "parent_id" in j else ""
                    entities[author_id] = entities.get(author_id, {"id" : author_id,
                                                                   "entity_type" : "user",
                                                                   "user_name" : j["author"],
                                                               })
                    entities[subreddit_id] = entities.get(subreddit_id, {"id" : subreddit_id,
                                                                         "entity_type" : "subreddit",
                                                                         "subreddit_name" : j["subreddit"]})

                    document = {
                        "id" : document_id,
                        "entity_type" : "submission" if "selftext" in j else "comment",
                        "response_to" : j.get("parent_id", "").split("_")[-1] if j.get("parent_id", 1) != j.get("link_id", 2) else "",
                        "for_submission" : j.get("link_id", "").split("_")[-1] if "link_id" in j else "",
                        "posted_in" : subreddit_id,
                        "score" : j["score"],
                        "gilded" : j["gilded"],
                        #"parent_id" : parent_id,
                        #"edited" : str(j["edited"]),
                        "text" : j["body"] if "body" in j else j["selftext"],
                        "title" : j["title"] if "title" in j else "",
                        "written_by" : author_id if "body" in j else "",
                        "submitted_by" : author_id if "selftext" in j else "",
                        "creation_time" : j["created_utc"],
                    }

                    document = {k : v for k, v in document.items() if v not in ["", None]}
                    #if document["entity_type"] == "submission":
                    ofd.write(json.dumps(document) + "\n")

        for eid, entity in entities.items():
            entity["id"] = eid
            entity = {k : v for k, v in entity.items() if v}
            ofd.write(json.dumps(entity) + "\n")
    
    
