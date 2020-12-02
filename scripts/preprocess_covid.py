import argparse
import gzip
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    count = 0
    entities = {"user" : {}, "tweet" : {}, "retweet" : {}, "quote" : {}}
    with gzip.open(args.inputs[0], "rt") as ifd:
        for line in ifd:
            count += 1
            if count > 5000:
                break
            j = json.loads(line)
            tweet_id = j["id_str"]
            user_id = j["user"]["screen_name"]
            tweet = {
                "text" : j["extended_tweet"]["full_text"] if "extended_tweet" in j else j["text"],
                "language" : j["lang"],
                "reply_count" : j["reply_count"],
                "source" : j["source"],
                "created_at" : j["created_at"],
                "tweeted_by" : user_id,
            }
            if "coordinates" in j and isinstance(j["coordinates"], dict) and "coordinates" in j["coordinates"]:
                lon, lat = j["coordinates"]["coordinates"]
                tweet["location"] = {"longitude" : lon, "latitude" : lat}
            entities["tweet"][tweet_id] = tweet
            user = {
                "name" : j["user"]["name"],
                "screen_name" : j["user"]["screen_name"],
                "description" : j["user"]["description"],
                "friends_count" : j["user"]["friends_count"],
                "followers_count" : j["user"]["followers_count"],
                "statuses_count" : j["user"]["statuses_count"],
                "favourites_count" : j["user"]["favourites_count"],
                "created_at" : j["user"]["created_at"],
            }
            entities["user"][user_id] = user
            for screen_name, time, source in j["retweeted"][0:10]:
                entities["user"][screen_name] = entities["user"].get(screen_name, {"screen_name" : screen_name})
                rt_id = "rt {}".format(len(entities["retweet"]))
                entities["retweet"][rt_id] = {
                    "source" : source,
                    "created_at" : time,
                    "retweet_of" : tweet_id,
                    "retweeted_by" : screen_name
                }
            for screen_name, time, source in j["quoted"][0:10]:
                entities["user"][screen_name] = entities["user"].get(screen_name, {"screen_name" : screen_name})
                q_id = "q {}".format(len(entities["quote"]))
                entities["quote"][q_id] = {
                    "source" : source,
                    "created_at" : time,
                    "quote_of" : tweet_id,
                    "quoted_by" : screen_name
                }

    with gzip.open(args.output, "wt") as ofd:
        for entity_type, es in entities.items():
            for eid, entity in es.items():
                entity = {k : v for k, v in entity.items() if v}
                entity["id"] = eid
                entity["entity_type"] = entity_type                
                ofd.write(json.dumps(entity) + "\n")
