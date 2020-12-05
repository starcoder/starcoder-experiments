import argparse
import logging
import gzip

logger = logging.getLogger("save_config")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config", help="Configuration")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--log_level", dest="log_level", default="ERROR", choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"], help="Logging level")
    args, rest = parser.parse_known_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(name)s - %(asctime)s - %(levelname)s - %(message)s'
    )
    
    with gzip.open(args.output, "wt") as ofd:
        ofd.write(args.config)
