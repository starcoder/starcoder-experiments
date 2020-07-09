import gzip
import pickle
import logging
import argparse
import torch
from starcoder.registry import field_model_classes, batchifier_classes, field_classes, scheduler_classes
from starcoder.ensemble import GraphAutoencoder
from starcoder.dataset import Dataset
from starcoder.schema import Schema, EncodedEntity, DecodedEntity
from starcoder.fields import CategoricalField
from starcoder.utils import stack_batch
import json

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data", help="Input data file")
    parser.add_argument("--split", dest="split", default=None, help="Data split (if none specified, run over everything)")
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--gpu", dest="gpu", default=False, action="store_true", help="Use GPU")
    parser.add_argument("--batch_size", dest="batch_size", default=128, type=int, help="")
    parser.add_argument("--log_level", dest="log_level", default="INFO", 
                        choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))

    with gzip.open(args.model, "rb") as ifd:
        state, margs, schema = torch.load(ifd)

    with gzip.open(args.data, "rb") as ifd:
        data = pickle.load(ifd)
    
    model = GraphAutoencoder(schema, 
                             margs.depth, 
                             margs.autoencoder_shapes,
                             reverse_relations=True,
    )
    model.load_state_dict(state)
    model.eval()
    
    if args.gpu:
        model.cuda()
        logging.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                     torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)

    model.train(False)

    if args.split == None:
        components = [i for i in range(data.num_components)]
    else:
        with gzip.open(args.split, "rb") as ifd:            
            indices = pickle.load(ifd)
            data = data.subselect_entities_by_index(indices)
            
    with gzip.open(args.output, "wt") as ofd:
        for i in range(data.num_components):
            component = data.component(i)
            entities, adjacencies = stack_batch([component], data.schema)
            reconstructed_entities, bottlenecks, _ = model(entities, adjacencies)
            for j in range(bottlenecks.shape[0]):
                original_entity, reconstructed_entity = {}, {}
                for field_name in entities.keys():
                    original_entity[field_name] = entities[field_name][j]
                    if field_name not in data.schema.relation_fields:                        
                        reconstructed_entity[field_name] = reconstructed_entities[field_name][j]

                original_entity = {k : v for k, v in data.schema.decode(EncodedEntity(original_entity)).items() if v not in [None]}
                reconstructed_entity = data.schema.decode(EncodedEntity(reconstructed_entity))
                entity = {"original" : original_entity,
                          "reconstruction" : {k : v for k, v in reconstructed_entity.items() if k in original_entity},
                          "bottleneck" : bottlenecks[j].tolist(),
                }
                ofd.write(json.dumps(entity) + "\n")
