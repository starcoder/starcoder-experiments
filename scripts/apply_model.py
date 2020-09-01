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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", dest="dataset", help="Input dataset file")
    parser.add_argument("--split", dest="split", default=None, help="Data split (if none specified, run over everything)")
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--output", dest="output", help="Output file")
    parser.add_argument("--gpu", dest="gpu", default=False, action="store_true", help="Use GPU")
    parser.add_argument("--batch_size", dest="batch_size", default=512, type=int, help="")
    parser.add_argument("--log_level", dest="log_level", default="INFO", 
                        choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))

    with gzip.open(args.model, "rb") as ifd:
        state, margs, schema = torch.load(ifd) # type: ignore

    with gzip.open(args.dataset, "rb") as ifd:
        dataset = pickle.load(ifd) # type: ignore
    
    model = GraphAutoencoder(schema, 
                             margs.depth, 
                             margs.autoencoder_shapes,
                             reverse_relationships=True,
    )
    model.load_state_dict(state)
    model.eval()
    
    if args.gpu:
        model.cuda()
        logging.info("CUDA memory allocated/cached: %.3fg/%.3fg", 
                     torch.cuda.memory_allocated() / 1000000000, torch.cuda.memory_cached() / 1000000000)

    model.train(False)
    if args.split == None:
        components = [i for i in range(dataset.num_components)]
    else:
        with gzip.open(args.split, "rb") as ifd:
            indices = pickle.load(ifd) # type: ignore
            dataset = dataset.subselect_entities_by_index(indices)
    
    num_batches = dataset.num_entities // args.batch_size
    num_batches = num_batches + 1 if num_batches * args.batch_size < dataset.num_entities else num_batches
    ids = dataset.ids
    batch_to_batch_ids = {b : [ids[i] for i in range(b * args.batch_size, (b + 1) * args.batch_size) if i < dataset.num_entities] for b in range(num_batches)}
    representation_storage = {}
    bottleneck_storage = {}
    logging.info("Dataset has %d entities", dataset.num_entities)

    reconstructions = {}
    for decoded_fields, norm, _ in apply_to_components(model, batchifier_classes["sample_components"]([]), dataset, args.batch_size, args.gpu):
        for entity in [dataset.schema.unpack(e) for e in unstack_entities(norm, dataset.schema)]:
            reconstructions[entity["id"]] = entity

    with open(args.output, "wt") as ofd:
        for eid, reconstruction in reconstructions.items():
            item = {"original" : dataset.entity(eid),
                    "reconstruction" : reconstructions[eid]}
            ofd.write(json.dumps(item) + "\n")
    logger.info("Saved reconstructions to '%s'", args.output)

    sys.exit()
    
    try:        
        for batch_num, batch_ids in batch_to_batch_ids.items():
            representation_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            bottleneck_storage[batch_num] = tempfile.mkstemp(prefix="starcoder")[1]
            packed_batch_entities: List[PackedEntity] = [schema.pack(data.entity(i)) for i in batch_ids]
            stacked_batch_entities = stack_entities(packed_batch_entities, data.schema.data_fields)
            encoded_batch_entities = model.encode_fields(stacked_batch_entities)
            entity_indices, field_indices, entity_field_indices = model.compute_indices(stacked_batch_entities)            
            encoded_entities = model.create_autoencoder_inputs(encoded_batch_entities, entity_indices)
            bottlenecks, outputs = model.run_first_autoencoder_layer(encoded_entities)
            torch.save((batch_ids, outputs, entity_indices), representation_storage[batch_num]) # type: ignore
            torch.save(bottlenecks, bottleneck_storage[batch_num]) # type: ignore
        for depth in range(1, model.depth + 1):
            bottlenecks = {}
            adjacencies: Adjacencies = {}
            bns = {}
            for batch_num, batch_ids in batch_to_batch_ids.items():
                for entity_type_name, bns in torch.load(bottleneck_storage[batch_num]).items(): # type: ignore
                    bottlenecks[entity_type_name] = bottlenecks.get(entity_type_name, torch.zeros(size=(data.num_entities, model.bottleneck_size)))
            for batch_num, batch_ids in batch_to_batch_ids.items():
                entity_ids, ae_inputs, entity_indices = torch.load(representation_storage[batch_num]) # type: ignore
                bottlenecks, outputs = model.run_structured_autoencoder_layer(depth, ae_inputs, bottlenecks, adjacencies, {}, entity_indices)
                
                torch.save((entity_ids, outputs, entity_indices), representation_storage[batch_num]) # type: ignore
                torch.save(bottlenecks, bottleneck_storage[batch_num]) # type: ignore

        with gzip.open(args.output, "wt") as ofd:            
            for batch_num, b_ids in batch_to_batch_ids.items():
                logging.info("Saving batch %d with %d entities", batch_num, len(b_ids))
                entity_ids, outputs, entity_indices = torch.load(representation_storage[batch_num]) # type: ignore
                bottlenecks = torch.load(bottleneck_storage[batch_num]) # type: ignore
                proj = torch.zeros(size=(len(b_ids), model.projected_size))                
                decoded_fields = model.decode_fields(model.project_autoencoder_outputs(outputs))
                decoded_entities = model.assemble_entities(decoded_fields, entity_indices)
                decoded_fields = {k : {kk : vv for kk, vv in v.items() if kk in data.schema.entity_types[k].data_fields} for k, v in decoded_fields.items()}
                ordered_bottlenecks = {}
                for entity_type, indices in entity_indices.items():
                    for i, index in enumerate(indices):
                        ordered_bottlenecks[index.item()] = bottlenecks[entity_type][i]

                for i, eid in enumerate(b_ids):
                    original_entity = data.entity(eid)                 
                    entity_type_name = original_entity[data.schema.entity_type_field.name]
                    entity_data_fields = data.schema.entity_types[entity_type_name].data_fields

                    reconstructed_entity = {data.schema.entity_type_field.name : entity_type_name}
                    for field_name in original_entity.keys():
                        if field_name in entity_data_fields:
                            reconstructed_entity[field_name] = decoded_entities[field_name][i].tolist()
                    reconstructed_entity = data.schema.unpack(reconstructed_entity)
                    entity = {"original" : original_entity,
                              # include missing-but-generated fields
                              "reconstruction" : {k : v for k, v in reconstructed_entity.items() if k in original_entity},
                              "bottleneck" : ordered_bottlenecks[i].tolist(),
                    }
                    ofd.write(json.dumps(entity) + "\n")
    except Exception as e:
        raise e
    finally:
        for tfname in list(bottleneck_storage.values()) + list(representation_storage.values()):
            try:
                os.remove(tfname)
            except Exception as e:
                logging.info("Could not clean up temporary file: %s", tfname)
