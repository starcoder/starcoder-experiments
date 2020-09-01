import argparse
import logging
import starcoder.dataset
from starcoder.registry import splitter_classes
import gzip
import random
import pickle

logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", dest="input", help="Input file")
    parser.add_argument("--splitter_class", dest="splitter_class", default="sample_components", choices=splitter_classes.keys(), help="Splitter class")
    #parser.add_argument("--proportions", dest="proportions", type=float, nargs="+", help="Proportions for splits")
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files for splits")
    parser.add_argument("--random_seed", dest="random_seed", default=None, type=int, help="Random seed")
    parser.add_argument("--log_level", dest="log_level", default="INFO", choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args, rest = parser.parse_known_args()
    splitter = splitter_classes[args.splitter_class](rest) #**vars(args))
    #setattr(args, "rest", rest)
    #print(splitter.proportions)
    
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    if isinstance(args.random_seed, int):
        logger.info("Setting random seed to %d across the board", args.random_seed)
        random.seed(args.random_seed)

    #assert(len(args.outputs) == len(args.proportions))
    
    with gzip.open(args.input, "rb") as ifd:
        data = pickle.load(ifd)
    logger.info("Loaded data set: %s", data)



    for fname, split in zip(args.outputs, splitter(data)):
        logger.info("Writing %d entities to %s", len(split), fname)
        with gzip.open(fname, "wt") as ofd:
            for i in split:
                ofd.write("{}\n".format(i))
                #pickle.dump(split, ofd)
