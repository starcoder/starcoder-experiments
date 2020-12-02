import sys
import gzip
import pickle
import logging
import argparse
import torch
from starcoder.registry import field_model_classes, batchifier_classes, field_classes, scheduler_classes
from starcoder.ensemble_model import GraphAutoencoder
from starcoder.dataset import Dataset
from starcoder.schema import Schema
from starcoder.field import CategoricalField
from starcoder.entity import stack_entities, UnpackedEntity, PackedEntity, Index, ID, unstack_entities
from starcoder.adjacency import Adjacencies
from starcoder.utils import apply_to_components
import json
import tempfile
import os
from typing import List, Type

logger = logging.getLogger(__name__)

def get_structure(model):
    return {k : get_structure(v) for k, v in model.named_children()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--log_level", dest="log_level", default="INFO", 
                        choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args, rest = parser.parse_known_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))

    with gzip.open(args.model, "rb") as ifd:
        state, margs, schema = torch.load(ifd) # type: ignore
    
    model = GraphAutoencoder(schema, 
                             margs.depth, 
                             margs.autoencoder_shapes,
                             reverse_relationships=True,
    )
    model.load_state_dict(state)
    structure = get_structure(model)
    with gzip.open(args.output, "wt") as ofd:
        ofd.write(json.dumps(structure, indent=4))
