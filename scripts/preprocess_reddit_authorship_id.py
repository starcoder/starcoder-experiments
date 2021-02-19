import argparse
import gzip
import json
import bz2

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()
    
    authors = {}
    subreddits = {}
    comments = {}
    submissions = {}
    with gzip.open(args.output, "wt") as ofd:
        for split, (submission_file, comment_file) in [
                ("train", args.inputs[0:2]),
                ("dev", args.inputs[2:4]),
                ("test", args.inputs[4:])
        ]:
            with bz2.open(submission_file, "rt") as ifd:
                for ln, line in enumerate(ifd):
                    if ln > 1000:
                        break
                    j = json.loads(line)
                    if "author" not in j:
                        continue
                    author_id = "{} author {}".format(split, j["author"])
                    subreddit_id = "{} subreddit {}".format(split, j["subreddit"])
                    authors[author_id] = {"author_name" : "{} {}".format(split, j["author"])}
                    subreddits[subreddit_id] = {"subreddit_name" : j["subreddit"]}
                    if "name" not in j:
                        j["name"] = "t3_{}".format(j["id"])
                    j["text"] = j["selftext"]
                    j["creation_time"] = int(j["created_utc"])
                    doc = {k : v for k, v in j.items() if k in ["title", "text", "creation_time"]} #"author" not in k and "subreddit" not in k}
                    doc["entity_type"] = "submission"
                    doc["id"] = "{} submission {}".format(split, j["name"])                    
                    doc["submitted_by"] = author_id
                    doc["posted_in"] = subreddit_id
                    doc = {k : v for k, v in doc.items() if v}
                    submissions[doc["id"]] = doc
                    #ofd.write(json.dumps(doc) + "\n")
            with bz2.open(comment_file, "rt") as ifd:
                for ln, line in enumerate(ifd):
                    if ln > 1000:
                        break
                    j = json.loads(line)
                    if "author" not in j:
                        continue
                    author_id = "{} author {}".format(split, j["author"])
                    subreddit_id = "{} subreddit {}".format(split, j["subreddit"])
                    authors[author_id] = {"author_name" : "{} {}".format(split, j["author"])}
                    subreddits[subreddit_id] = {"subreddit_name" : j["subreddit"]}
                    if "name" not in j:
                        j["name"] = "t1_{}".format(j["id"])
                    j["text"] = j["body"]
                    j["creation_time"] = int(j["created_utc"])
                    doc = {k : v for k, v in j.items() if k in ["edited", "text", "creation_time"]}
                    doc["entity_type"] = "comment"
                    doc["id"] = "{} comment {}".format(split, j["name"])
                    doc["written_by"] = author_id
                    if j["parent_id"] != j["link_id"]:
                        doc["response_to"] = "{} comment {}".format(split, j["parent_id"])                    
                    doc["for_submission"] = "{} submission {}".format(split, j["link_id"])
                    doc["posted_in"] = subreddit_id
                    doc = {k : v for k, v in doc.items() if v}
                    comments[doc["id"]] = doc
                    #ofd.write(json.dumps(doc) + "\n")
        for k, v in authors.items():
            v["id"] = k
            v["entity_type"] = "author"
            v = {k : vv for k, vv in v.items() if vv}
            ofd.write(json.dumps(v) + "\n")
        for k, v in subreddits.items():
            v["id"] = k
            v["entity_type"] = "subreddit"
            v = {k : vv for k, vv in v.items() if vv}
            ofd.write(json.dumps(v) + "\n")
        for k, v in submissions.items():
            v["id"] = k
            v["entity_type"] = "submission"
            v = {k : vv for k, vv in v.items() if vv}
            ofd.write(json.dumps(v) + "\n")
        for k, v in comments.items():
            v["id"] = k
            v["entity_type"] = "comment"
            if v["for_submission"] not in submissions:
                del v["for_submission"]
            if "response_to" in v and v["response_to"] not in comments:
                del v["response_to"]
            v = {k : vv for k, vv in v.items() if vv}
            ofd.write(json.dumps(v) + "\n")
