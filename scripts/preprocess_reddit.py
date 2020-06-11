import argparse
import gzip
import bz2
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="inputs", help="Input files")
    parser.add_argument("-s", "--schema", dest="schema", help="Schema file")    
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    entities = {}
    with gzip.open(args.output, "wt") as ofd:
        for fname in args.inputs:
            with bz2.open(fname, "rt") as ifd:
                for i, line in enumerate(ifd):
                    j = json.loads(line)
                    author_id = j["author"]
                    subreddit_id = j["subreddit_id"]
                    comment_id = j["name"]
                    parent_id = j["parent_id"]

                    entities[author_id] = entities.get(author_id, {"id" : author_id,
                                                                   "entity_type" : "author",
                                                                   "author_name" : "author_id",
                                                                   "wrote" : []})
                    entities[subreddit_id] = entities.get(subreddit_id, {"id" : subreddit_id,
                                                                         "entity_type" : "subreddit",
                                                                         "subreddit_name" : j["subreddit"]})
                    comment = {
                        "id" : comment_id,
                        "entity_type" : "comment",
                        "response_to" : j["parent_id"],
                        "posted_in" : subreddit_id,
                        "upvotes" : j["ups"],
                        "downvotes" : j["downs"],
                        "gilded" : j["gilded"],
                        "edited" : str(j["edited"]),
                        "text" : j["body"],
                    }
                    ofd.write(json.dumps(comment) + "\n")
                    
                    entities[author_id]["wrote"].append(comment_id)
                    entities[author_id]["author_name"] = j["author"]

        for eid, entity in entities.items():
            entity["id"] = eid
            ofd.write(json.dumps(entity) + "\n")
    
    
