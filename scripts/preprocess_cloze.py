import argparse
import zipfile
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input file")
    parser.add_argument("-o", "--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    entities = {k : [] for k in ["story", "line", "character", "annotator", "annotation"]}
    with zipfile.ZipFile(args.inputs[0], "r") as izfd:
        with izfd.open("json_version/annotations.json", "r") as ifd:
            for story_id, story in json.loads(ifd.read()).items():
                part = story["partition"]
                title = story["title"]
                for line_num, line in story["lines"].items():
                    sentence = line["text"]
                    # emotion motiv
                    # plutchik
                    # maslow reiss text

