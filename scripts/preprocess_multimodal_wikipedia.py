import json
import argparse
import zipfile
import re
import pickle
import sys
import os.path
import gzip
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(nargs="+", dest="inputs", help="Input files")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    data_path = os.path.dirname(args.inputs[0])
    with gzip.open(args.inputs[0], "rt") as ifd, gzip.open(args.output, "wt") as ofd:
        for i, line in enumerate(ifd):
            j = json.loads(line)
            #for field_name in ["has_image", "image"]:
            #    if field_name in j:
            #        del j[field_name]
            if j["entity_type"] == "article" or os.path.exists(os.path.join(data_path, "full_media", j[j["entity_type"]])):
                ofd.write(json.dumps(j) + "\n")                
                #if 
            #     pass
            #     #j["text"] = j["text"][:10]
            # elif j["entity_type"] == "video":
            #     try:
            #         with gzip.open(os.path.join(data_path, "downsampled", j["video"]), "rb") as m_ifd:
            #             a, v = pickle.load(m_ifd)
            #             j["video"] = [a.tolist(), v.tolist()] #m.tolist()
            #             #print("video ({}, {})".format(a.shape, v.shape))
            #     except FileNotFoundError as e:                        
            #         #except Exception as e:
            #         #print("skipping image {}".format(j["image"]))                    
            #         #print(j["video"])
            #         del j["video"]
            #         #raise e
            #     pass
            # elif j["entity_type"] == "image":
            #     try:
            #         with gzip.open(os.path.join(data_path, "downsampled", j["image"]), "rb") as m_ifd:
            #             im = pickle.load(m_ifd)
            #             j["image"] = im.tolist()
            #             #print("image ({})".format(im.shape))
            #     except FileNotFoundError as e:
                    
            #         #print(j["id"])
            #         #print("skipping image {}".format(j["image"]))
            #         del j["image"]
            #         #raise e
            #     pass
            # elif j["entity_type"] == "audio":
            #     #print(j["audio"])
            #     try:
            #         with gzip.open(os.path.join(data_path, "downsampled", j["audio"]), "rb") as m_ifd:
            #             m = pickle.load(m_ifd)
            #             j["audio"] = m.tolist()
            #             #print("audio ({})".format(m.shape))
            #     except FileNotFoundError as e:
            #         del j["audio"]
            #     except Exception as e:
            #         print(j)
            #         raise e
            # else:
            #     continue    
            #ofd.write(json.dumps(j) + "\n")
