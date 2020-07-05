import argparse
import logging
import starcoder.dataset
from starcoder.registry import sampler_classes
import gzip
import random
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--proportions", dest="proportions", type=float, nargs="+", help="Proportions for splits")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files for splits")
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int, help="Random seed")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    parser.add_argument("--split_level", dest="split_level", default="entities", choices=["entities", "components"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if isinstance(args.random_seed, int):
        logging.info("Setting random seed to %d across the board", args.random_seed)
        random.seed(args.random_seed)

    assert(len(args.outputs) == len(args.proportions))
    
    with gzip.open(args.input, "rb") as ifd:
        data = pickle.load(ifd)
    logging.info("Loaded data set: %s", data)

    
    
    if args.split_level == "components":
        indices = list(range(data.num_components))
        #indices = sum([data.component_indices(i) for i in range(data.num_components)], [])
    elif args.split_level == "entities":
        indices = list(range(len(data)))
    
    random.shuffle(indices)

    total = sum(args.proportions)
    splits = [(f, int((p / total) * len(indices))) for f, p in zip(args.outputs, args.proportions)]

    for fname, c in splits:
        logging.info("Writing %d %s to %s", c, args.split_level, fname)
        with gzip.open(fname, "wb") as ofd:
            if args.split_level == "components":
                pickle.dump(sum([data.component_indices(i) for i in indices[:c]], []), ofd)
            elif args.split_level == "entities":
                pickle.dump(indices[:c], ofd)                
            indices = indices[c:]
