import argparse
import gzip
import json
                 
import argparse
import pandas
import gzip
import json
import sys
import numpy

bloomberg_fields = {k : v for k, v in [
    ("codeID", "coder_id"),
    ("support", "pro_government"),
    ("opposed", "anti_government"),
] + [(x, x) for x in [
    'health',
    'riskinc',
    'risklow',
    'mcm',
    'fatal',
    'spread',
    'reduction',
    'travelban',
    'quarantine',
    'screening',
    'phmonitor',
    'refute',
    'rumormentioned',
    'rumor',
    'rumor2',
    'hashtag',
    'hashtag2',
    'othhashtag',
    'inaccessible',
    'totalmention',
    'opfact',
    'riskcombine',
    'truthcombine',
    'rumor_num',
    "excluded",
    "joke",
    "headline",
    "true",
    "halftrue",
    "false",
    "unknown",
    "opinion",
    "discord",
    "political",
]]
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    annotations, excel = args.inputs

    entities = {"user" : {}, "tweet" : {}}
    
    tweets = {}
    anns = pandas.read_stata(annotations)
    anns.codenumber = anns.codenumber.apply(int)    

    ex = pandas.read_excel(excel)

    both = anns.merge(ex, left_on="codenumber", right_on="Code #")
    for i in range(both.shape[0]):
        row = both.iloc[i]
        tweet_id = str(row["tweet_id"])
        user_id = str(row["user_screen_name"])
        try:
            user = {
                "screen_name" : user_id,
            }
            if isinstance(row["user_name"], str):
                user["name"] = row["user_name"]
            entities["user"][user_id] = user
            tweet = {
                "text" : str(row["text"]),
                "created_at" : str(row["created_at"]),
                "retweet_count" : int(row["retweet_count"]),
                "favorite_count" : int(row["favorite_count"]),
                "tweeted_by" : user_id,
            }
            for old_name, new_name in bloomberg_fields.items():
                continue
                v = row[old_name]
                if v != "":                
                    tweet[new_name] = float(v) if isinstance(v, (numpy.int8, numpy.float32)) else v               
            entities["tweet"][tweet_id] = tweet
        except:
            continue
    with gzip.open(args.output, "wt") as ofd:
        for entity_type, es in entities.items():
            for eid, entity in es.items():
                entity = {k : v for k, v in entity.items() if v}
                entity["id"] = eid
                entity["entity_type"] = entity_type
                ofd.write(json.dumps(entity) + "\n")
