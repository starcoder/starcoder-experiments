import argparse
import tarfile
import gzip
import json
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="inputs", help="Input files")    
    parser.add_argument("-s", "--schema", dest="schema", help="Schema file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args = parser.parse_args()

    all_fluencies = set()
    count = 0
    with gzip.open(args.output, "wt") as ofd:
        for file_id, fname in list(enumerate(args.inputs)):
            try:
                with gzip.open(fname, "rb") as ifd:
                    main_user, followed_by, user_tweets = pickle.load(ifd)
                    count += 1
                    main_id = "{}_{}".format(file_id, main_user["id"])
                    fluencies= [("fluent_in_{}".format(l), "true") for l in main_user["fluencies"].keys() if l != "bot"]
                    is_bot = main_user.get("labels", {}).get("bot") == '1' or "bot" in main_user.get("fluencies", {})
                    #if is_bot:
                    #    continue
                    entities = {main_id : {k : v for k, v in [("entity_type", "user"), ("followed_by", []), ("follows", [])] + \
                                           fluencies + [("is_bot", "true" if is_bot else "false")]}}
                    for f, _ in fluencies:
                        all_fluencies.add(f)
                    for followed, follower in followed_by:
                        if followed == -1:
                            followed = main_id
                        else:
                            followed = "{}_{}".format(file_id, followed)
                        if followed != main_id:
                            continue
                        follower = "{}_{}".format(file_id, follower)
                        entities[followed] = entities.get(followed, {"entity_type" : "user", "followed_by" : []})
                        entities[follower] = entities.get(follower, {"entity_type" : "user", "followed_by" : []})
                        entities[followed]["followed_by"].append(follower)

                    for user_id, tweets in user_tweets.items():
                        user_id = "{}_{}".format(file_id, user_id)
                        if user_id in entities:
                            entities[user_id]["wrote"] = []
                            for tweet_id, tweet in tweets.items():
                                if sum(tweet["valid"].values()) == 0.0:
                                    continue
                                tweet_id = "{}_{}".format(file_id, tweet_id)
                                entities[tweet_id] = {"twitter_language" : list(tweet["twitter"][1].keys())[0],
                                                      "valid_vector" : tweet["valid"], #[v for k, v in sorted(tweet["valid"].items())],
                                                      "entity_type" : "tweet",
                                                      "written_by" : user_id,
                                                      "text" : tweet["text"],
                                }
                    for eid, entity in entities.items():
                        entity["id"] = eid
                        for f in all_fluencies:
                            if f not in entity:
                                entity[f] = "false"
                        try:
                            ofd.write(json.dumps(entity) + "\n")
                        except Exception as e:
                            print(entity)
                            raise e
            except EOFError:
                pass
