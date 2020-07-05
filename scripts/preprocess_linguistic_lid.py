import argparse
import gzip
import json
import csv
import pycountry
import os.path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("-o", "--output_file", dest="output_file", help="Output file")
    args = parser.parse_args()

    wals_langs = set()
    tweet_langs = set()
    keys = ["family", 
            #"iso_code", 
            #"wals_code", 
            #"countrycodes", 
            "macroarea", 
            #"Name", 
            #"glottocode", 
            "genus", 
            #"141A Writing Systems"
    ]
    languages = {}
    genuses = {}
    families = {}
    macroareas = {}
    properties_file = [fname for fname in args.inputs if os.path.basename(fname).startswith("wals")][0]
    tweet_file = [fname for fname in args.inputs if fname != properties_file][0]
    
    with gzip.open(properties_file, "rt") as ifd:
        for row in csv.DictReader(ifd):
            iso_code = row["iso_code"]
            lang = pycountry.languages.get(alpha_3=iso_code)
            if hasattr(lang, "alpha_2"):
                lang_code = lang.alpha_2
                wals_langs.add(lang_code)
                macroareas[row["macroarea"]] = {"id" : row["macroarea"],
                                               "entity_type" : "macroarea",
                                               "macroarea_name" : row["macroarea"]}
                families[row["family"]] = {"id" : row["family"],
                                           "entity_type" : "family",
                                           "from_macroarea" : row["macroarea"],
                                           "family_name" : row["family"]}
                genuses[row["genus"]] = {"id" : row["genus"], 
                                         "from_family" : row["family"], 
                                         "entity_type" : "genus",
                                         "genus_name" : row["genus"]
                }
                languages[lang_code] = {"id" : lang_code,
                                        "entity_type" : "language",
                                        "from_genus" : row["genus"],
                                        "language_name" : row["Name"]
                }

    tweets = []
    with gzip.open(tweet_file, "rt") as ifd:
        for line in ifd:
            tid, lang_code, text = line.strip().split("\t")
            tweet_langs.add(lang_code)
            tweets.append({"id" : tid,
                           "entity_type" : "tweet",
                           "written_in" : lang_code,
                           "tweet_text" : text.lower(),
                           "tweet_language" : lang_code,
            })

    slanguages, sgenuses, sfamilies, smacroareas = set(), set(), set(), set()
    with gzip.open(args.output_file, "wt") as ofd:
        for tweet in tweets:
            lang = tweet["written_in"]
            if lang in wals_langs:
                ofd.write(json.dumps(tweet) + "\n")
                slanguages.add(lang)
                sgenuses.add(languages[lang]["from_genus"])
                sfamilies.add(genuses[languages[lang]["from_genus"]]["from_family"])
                smacroareas.add(families[genuses[languages[lang]["from_genus"]]["from_family"]]["from_macroarea"])
        for language in slanguages:
            ofd.write(json.dumps(languages[language]) + "\n")
        for genus in sgenuses:
            ofd.write(json.dumps(genuses[genus]) + "\n")
        for family in sfamilies:
            ofd.write(json.dumps(families[family]) + "\n")            
        for macroarea in smacroareas:
            ofd.write(json.dumps(macroareas[macroarea]) + "\n")
