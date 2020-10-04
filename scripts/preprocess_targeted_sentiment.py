import argparse
import gzip
import json
from nltk import Tree

# https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

def process_tree(node, path, split, top_level=False):
    parent_id = "_".join([str(x) for x in path[:-1]])
    node_id = "_".join([str(x) for x in path])
    entity = {"id" : node_id,
              "split" : split}

    if isinstance(node, str):
        entity["text"] = node
        entity["entity_type"] = "word"
        entity["text_of"] = parent_id
        return [entity]
    else:
        entity["entity_type"] = "node"
        entity["sentiment"] = int(node.label())
        if not top_level:
            entity["child_of"] = parent_id
        return [entity] + sum([process_tree(c, path + [i], split) for i, c in enumerate(node)], [])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(dest="inputs", nargs="+", help="Input files")
    parser.add_argument("--output", dest="output", help="Output file")
    args, rest = parser.parse_known_args()

    trees = []
    for fname in args.inputs:
        split = "train" if "train" in fname else "dev" if "dev" in fname else "test" if "test" in fname else None
        assert split != None
        with gzip.open(fname, "rt") as ifd:
            for i, line in enumerate(ifd):
                tree = Tree.fromstring(line)
                path = [split, i]
                node_id = "_".join([str(x) for x in path])
                trees += (process_tree(tree, path + [0], split, top_level=True))

    with gzip.open(args.output, "wt") as ofd:
        ofd.write("\n".join([json.dumps(tree) for tree in trees]))
