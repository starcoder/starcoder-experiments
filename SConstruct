import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json
from hashlib import md5
import steamroller


# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

def camel_case(name):
    return name.replace("_", " ").title().replace(" ", "")

vars = Variables("custom.py")
vars.AddVariables(
    ("OUTPUT_WIDTH", "", 100),
    ("RANDOM_COMPONENTS", "", 1000),
    ("MINIMUM_CONSTANTS", "", 1),
    ("MAXIMUM_CONSTANTS", "", 4),
    ("EXTRA_ARGS", "", ""),
    ("TRAIN_PROPORTION", "", 0.9),
    ("DEV_PROPORTION", "", 0.1),
    ("MAX_CATEGORICAL", "", 50),
    ("MAX_SEQUENCE_LENGTH", "", 100),
    ("RANDOM_LINE_COUNT", "", 1000),
    ("MAX_COLLAPSE", "", 0),
    ("MAX_SEQUENCE_LENGTH", "", 32),
    ("LOG_LEVEL", "", "INFO"),
    ("LINE_COUNT", "", 1000),
    ("BATCH_SIZE", "", 128),
    ("MAX_EPOCHS", "", 10),
    ("LEARNING_RATE", "", 0.001),
    ("RANDOM_RESTARTS", "", 0),
    ("ELASTIC_HOST", "", "localhost"),
    ("ELASTIC_PORT", "", 5601),
    ("MOMENTUM", "", None),
    ("EARLY_STOP", "", 10),
    ("PATIENCE", "", 5),
    ("HIDDEN_SIZE", "", 32),
    ("EMBEDDING_SIZE", "", 32),
    ("AUTOENCODER_SHAPES", "", (1024, 512)),
    ("CLUSTER_REDUCTION", "", 0.5),
    BoolVariable("USE_GPU", "", False),
    BoolVariable("USE_GPU_APPLY", "", False),
    BoolVariable("USE_GRID", "", False),
    ("GPU_PREAMBLE", "", "module load cuda11.0/toolkit"),
    ("GPU_QUEUE", "", "gpu.q"),
    ("SPLIT_PROPORTIONS", "", (("train", 0.80), ("dev", 0.10), ("test", 0.10))),
    ("DEPTH", "", 1),
    ("SPLITTER_CLASS", "", "sample_components"),
    ("BATCHIFIER_CLASS", "", "sample_components"),
    ("DATA_PATH", "", ""),
    ("EXPERIMENTS", "", {}),
    ("LOCATION_CACHE", "", None),
    ("TRAIN_FIELD_DROPOUT", "", 0.1),
    ("TRAIN_NEURON_DROPOUT", "", 0.1),
    ("DEV_FIELD_DROPOUT", "", 0.1),
    ("TEST_FIELD_DROPOUT", "", 1.0),
    ("CACHED", "", True),
)

env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", steamroller.generate])


preprocessors = {}
for exp_name, exp_spec in env["EXPERIMENTS"].items():
    preprocessors["preprocess_{}".format(exp_name)] = env.Builder(
        **env.ActionMaker(
            "python",
            "scripts/preprocess_{}.py".format(exp_name),
            "${' '.join([\"'%s'\" % (s) for s in SOURCES])} --output ${TARGETS[0]} ${'--location_cache ' if LOCATION_CACHE != None else ''} ${LOCATION_CACHE} ${EXTRA_ARGS}"
        )
    )
env.Append(BUILDERS=preprocessors)
    
env.Append(
    BUILDERS={
        "PrepareDataset" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/prepare_dataset.py",
                "--schema_output ${TARGETS[0]} --data_output ${TARGETS[1]} --data_input ${SOURCES[0]} --schema_input ${SOURCES[1]}",
                other_deps=[]
            )
        ),
        "SplitData" : env.Builder(
            **env.ActionMaker(
                "python",
                "",
                "-m starcoder.splitter --schema ${SOURCES[0]} --input ${SOURCES[1]} --random_seed ${RANDOM_SEED} ${TARGETS}",
                other_deps=[],
            )
        ),
        "TrainModel" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/train_model.py",
                "--schema ${SOURCES[0]} --data ${SOURCES[1]} --train ${SOURCES[2]} --dev ${SOURCES[3]} --model_output ${TARGETS[0]} ${'--gpu' if USE_GPU else ''}",
                USE_GPU=env["USE_GPU"],
            ),
            USE_GPU=env["USE_GPU"],
        ),
        "ApplyModel" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/apply_model.py",
                "--schema ${SOURCES[0]} --model ${SOURCES[1]} --dataset ${SOURCES[2]} ${'--split ' + SOURCES[3].rstr() if len(SOURCES) == 4 else ''} --output ${TARGETS[0]} ${'--blind' if BLIND else ''} ${'--remove_structure' if REMOVE_STRUCTURE else ''} --log_level ${LOG_LEVEL} ${'--cached ' if CACHED else ''}",
                USE_GPU=env["USE_GPU"],
            ),
            USE_GPU=env["USE_GPU"]
        ),
        "TopicModel" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/train_topic_model.py",
                "--data ${SOURCES[0]} --schema ${SOURCES[1]} --output ${TARGETS[0]}",
            )
        ),
        "MakeTSNE" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/make_tsne.py",
                "--schema ${SOURCES[0]} --data ${SOURCES[1]} --output ${TARGETS[0]}",
            )
        ),
        "LIWC" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/apply_liwc.py",
                "--data ${SOURCES[0]} --schema ${SOURCES[1]} --liwc ${DATA_PATH}/liwc/liwc.json --output ${TARGETS[0]}",
            )
        ),
        "Decameron" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/preprocess_decameron.py",
                "${SOURCES[0]} --schema ${TARGETS[0]} --data ${TARGETS[1]}",
            )
        ),
        "ExpandSchema" : env.Builder(
            **env.ActionMaker(
                "python",
                "",
                "-m starcoder.schema --schema ${SOURCES[0]} ${SOURCES[1:]} --output ${TARGETS[0]}",
            )
        ),
        "SaveConfig" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/save_config.py",
                "--output ${TARGETS[0]} --config '${CONFIG}'",
            )
        ),
        "ExtractModelStructure" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/extract_model_structure.py",
                "--model ${SOURCES[0]} --output ${TARGETS[0]}",
            )
        ),
        "CollateOutputs" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/collate_outputs.py",
                "${SOURCES[1:]} --output ${TARGETS[0]} ${'--test ' + SOURCES[0].rstr() if LIMIT_TO_TEST==True else ''}",
            )
        ),
        "VisualizeImages" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/visualize_images.py",
                "--input ${SOURCES[0]} ${'--id_file ' + SOURCES[1].rstr() if len(SOURCES) == 2 else ''} --output ${TARGETS[0]}"
            )
        ),
    },
    tools=["default"],
)

def expand_cells(grid_search):
    retval = [[]]
    for pat, parvals in grid_search:
        for par, vals in parvals.items():
            retval = sum([[old + [(pat, {par : val})] for val in vals] for old in retval], [])    
    return retval

# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)


# and the command-printing function
env['PRINT_CMD_LINE_FUNC'] = print_cmd_line


# and how we decide if a dependency is out of date
def decider(dependency, target, prev_ni, repo_node):    
    try:
        prev_ts = getattr(prev_ni, "timestamp")
    except:
        prev_ts = None
    if dependency.get_timestamp() != prev_ts or not dependency.is_up_to_date():
        #target.set_timestamp(dependency.get_timestamp())
        return True
    return False
env.Decider("timestamp-newer") #decider)


def expand_configuration(*configs):
    unexpanded = {}
    for config in configs:
        for k, v in config.items():          
            unexpanded[k] = v if isinstance(v, list) else [v]
    expanded = [[]]
    for arg_name, values in unexpanded.items():
        expanded = sum(
            [
                [
                    config + [
                        (
                            arg_name.upper(), 
                            v
                        )
                    ] for config in expanded
                ] for v in values
            ], 
            []
        )
    return [{k : v for k, v in c} for c in expanded]

def run_experiment(env, experiment_config, **args):
    outputs = []
    blind_outputs = []
    experiment_name = args["EXPERIMENT_NAME"]
    pats = experiment_config.get("DATA_FILES", [])
    data = sum([env.Glob(env.subst(p)) for p in (pats if isinstance(pats, (list, tuple)) else [pats])], [])
    if experiment_name != "decameron":
        initial_schema = experiment_config.get("SCHEMA", None)
        data = getattr(env, "preprocess_{}".format(experiment_name))("work/${EXPERIMENT_NAME}/data.json.gz",
                                                                     data, **args, **experiment_config)
    else:
        initial_schema, data = env.Decameron(
            ["work/${EXPERIMENT_NAME}/auto_schema.json", "work/${EXPERIMENT_NAME}/data.json.gz"],
            data, **args, **experiment_config)

    schema = env.ExpandSchema(
        "work/${EXPERIMENT_NAME}/schema.json",
        [
            initial_schema, 
            "configurations/default.json"
        ] + env.Glob("configurations/{}.json".format(experiment_name)),
        **args, **experiment_config
    )

    env.Alias("schemas", schema)
    env.Alias("data", data)
    
    tm = env.TopicModel(
        "work/${EXPERIMENT_NAME}/topic_models.json.gz",
        [data, schema],
        **args
    )
    env.Alias("topics", tm)

    liwc = env.LIWC(
        "work/${EXPERIMENT_NAME}/liwc.json.gz",
        [data, schema],
        **args
    )
    env.Alias("liwc", liwc)
    
    train, dev, test = env.SplitData(["work/${{EXPERIMENT_NAME}}/{0}_ids.json.gz".format(n) for n in ["train", "dev", "test"]], 
                                     [schema, data],                                     
                                     **experiment_config,
                                     **args, RANDOM_SEED=0)
    env.Alias("splits", [train, dev, test])

    for i, cell in enumerate(expand_cells(experiment_config.get("GRID_SEARCH", []))):
        cell = experiment_config.get("OVERRIDES", []) + cell
        cell = shlex.quote(str(cell)) if env["USE_GRID"] else str(cell)
        cell_schema = env.ExpandSchema(
            "work/${EXPERIMENT_NAME}/schema_${CELL_INDEX}.json",
            [
                schema, 
                env.Value(cell),
            ],
            CELL_INDEX=i,
            **args, 
            **experiment_config
        )
        env.Alias("schemas", cell_schema)
        model = env.TrainModel("work/${EXPERIMENT_NAME}/model_${CELL_INDEX}.pkl.gz",
                               [cell_schema, data, train, dev],
                               CELL_INDEX=i,
                               **experiment_config,
                               **args
        )
        env.Alias("models", model)

        output = env.ApplyModel("work/${EXPERIMENT_NAME}/output_${CELL_INDEX}.json.gz",
                                [cell_schema, model, data],
                                CELL_INDEX=i,                                
                                **{k : (False if k == "REMOVE_STRUCTURE" else v) for k, v in experiment_config.items()},
                                **args,
                            )
        env.Alias("outputs", output)
        outputs.append((cell_schema, output))
        
        blind_output = env.ApplyModel("work/${EXPERIMENT_NAME}/blind_output_${CELL_INDEX}.json.gz",
                                      [cell_schema, model, data],
                                      CELL_INDEX=i,
                                      BLIND=True,
                                      **experiment_config,
                                      **args
                                  )
        env.Alias("blind_outputs", blind_output)
        blind_outputs.append((cell_schema, blind_output))

        tsne = env.MakeTSNE("work/${EXPERIMENT_NAME}/tsne_${CELL_INDEX}.json.gz",
                            [cell_schema, output],
                            CELL_INDEX=i,
                            **experiment_config,
                            **args
                        )

        if experiment_name == "death_row":
            img = env.VisualizeImages("work/${EXPERIMENT_NAME}/image_reconstructions_${CELL_INDEX}.png", 
                                      output,
                                      CELL_INDEX=i,
                                      **experiment_config,
                                      **args
            )
            test_img = env.VisualizeImages("work/${EXPERIMENT_NAME}/test_image_reconstructions_${CELL_INDEX}.png", 
                                           [output, test],
                                           CELL_INDEX=i,
                                           **experiment_config,
                                           **args
            )

    blind_results = env.CollateOutputs(
        "work/${EXPERIMENT_NAME}/blind_results.csv", 
        [test] + blind_outputs,
        **experiment_config,
        **args,
        LIMIT_TO_TEST=True
    )
    env.Alias("blind_results", blind_results)

    results = env.CollateOutputs(
        "work/${EXPERIMENT_NAME}/results.csv", 
        [test] + outputs,
        **experiment_config,
        **args,
        LIMIT_TO_TEST=False
    )
    env.Alias("results", results)
    
    return None


env.AddMethod(run_experiment, "RunExperiment")


#
# Run all experiments
#
models = []
names = []
for experiment_name, experiment_config in env["EXPERIMENTS"].items():
    names.append(experiment_name)
    models.append(env.RunExperiment(experiment_config, EXPERIMENT_NAME=experiment_name))
