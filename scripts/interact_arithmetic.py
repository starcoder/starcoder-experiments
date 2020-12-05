import sys
import gzip
import logging
import argparse
import torch
from starcoder.dataset import Dataset
from starcoder.property import NumericProperty, DistributionProperty, IdProperty, EntityTypeProperty, RelationProperty
from starcoder.models import GraphAutoencoder
from arithmetic import render_tree, flatten_tree, grow_tree, reverse_edges, populate_tree, to_json
from starcoder.utils import batchify, batch_to_list
import ast
import cmd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", help="Model file")
    parser.add_argument("--log_level", dest="log_level", default="INFO", 
                        choices=["ERROR", "WARNING", "INFO", "DEBUG"], help="Logging level")
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level))

    with gzip.open(args.model, "rb") as ifd:
        state, margs, schema = torch.load(ifd)

    model = GraphAutoencoder(schema, 
                             margs.depth, 
                             margs.autoencoder_shapes,
                             embedding_size=margs.embedding_size,
                             hidden_size=margs.hidden_size,
                             property_dropout=margs.property_dropout,
                             hidden_dropout=margs.hidden_dropout,
                             reverse_relations=True
    )
    model.load_state_dict(state)
    model.eval()
    model.train(False)
    logging.debug("Model: %s", model)
    
    def one_equation(eq):
        entities = to_json(populate_tree(ast.parse(eq).body[0], "0"))
        correct = {}
        num_consts = 0
        for i in range(len(entities)):
            correct[entities[i]["id"]] = entities[i]["value"]
            if entities[i]["operation_name"] != "const":            
                del entities[i]["value"]
            else:
                num_consts += 1
                
        data = Dataset(schema, entities)        
        for i, ((fents, fadjs), (ments, madjs)) in enumerate(batchify(data, len(entities), subselect=False)):
            #print(num_consts)
            #model._depth = num_consts - 1
            reconstructions, bottlenecks, _ = model(ments, madjs)
            reconstructions = {k : (v if k == schema.entity_type_property.name or k == schema.id_property.name or isinstance(schema.properties[k], (DistributionProperty, NumericProperty, IdProperty, EntityTypeProperty, Relationship))
                                    else torch.argmax(v, -1, False)) for k, v in reconstructions.items() if k not in schema.relationships}
            for entity in [schema.decode(b) for b in batch_to_list(reconstructions)]:
                print("Entity {} should be {:.3f}, got {:.3f}".format(entity["id"], correct[entity["id"]], entity["value"]))

    class Interp(cmd.Cmd):
        def default(self, line):
            if line.strip().lower().startswith("q"):
                return True
            else:
                try:
                    one_equation(line)
                except Exception as e:
                    raise(e)
                    #print("I had trouble understanding '{}'".format(line.strip()))

    interp = Interp()
    interp.cmdloop("Enter equations consisting of numbers, '+', '-', '*', '/', and parentheses  ('q' to exit)")
