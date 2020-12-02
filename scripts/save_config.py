import argparse
import gzip

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="Configuration")
    parser.add_argument("--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()
    with gzip.open(args.output, "wt") as ofd:
        ofd.write(args.config)
