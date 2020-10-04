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
    ("AUTOENCODER_SHAPES", "", [32, 16]),
    ("CLUSTER_REDUCTION", "", 0.5),
    BoolVariable("USE_GPU", "", False),
    BoolVariable("USE_GRID", "", False),
    ("GPU_PREAMBLE", "", "module load cuda10.1/toolkit"),
    ("SPLIT_PROPORTIONS", "", [("train", 0.80), ("dev", 0.10), ("test", 0.10)]),
    ("DEPTH", "", 1),
    ("SPLITTER_CLASS", "", "sample_components"),
    ("BATCHIFIER_CLASS", "", "sample_components"),
    ("DATA_PATH", "", ""),
    ("EXPERIMENTS", "", {}),
    ("LOCATION_CACHE", "", None),    
)

env = Environment(variables=vars, ENV=os.environ, TARFLAGS="-c -z", TARSUFFIX=".tgz",
                  tools=["default", steamroller.generate])

preprocessors = {}
for exp_name, exp_spec in env["EXPERIMENTS"].items():
    preprocessors["preprocess_{}".format(exp_name)] = env.Builder(
        **env.ActionMaker(
            "python",
            "scripts/preprocess_{}.py".format(exp_name),
            "${SOURCES} --output ${TARGETS[0]} ${'--location_cache ' if LOCATION_CACHE != None else ''} ${LOCATION_CACHE}"
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
                "scripts/split_data.py",
                "--input ${SOURCES[0]} --proportions ${PROPORTIONS} --outputs ${TARGETS} --random_seed ${RANDOM_SEED} --splitter_class ${SPLITTER_CLASS} --shared_entity_types ${SHARED_ENTITY_TYPES}",
                other_deps=[],
            )
        ),
        "TrainModel" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/train_model.py",
                "--data ${SOURCES[0]} --train ${SOURCES[1]} --dev ${SOURCES[2]} --model_output ${TARGETS[0]} --trace_output ${TARGETS[1]} ${'--gpu' if USE_GPU else ''} ${'--autoencoder_shapes ' + ' '.join(map(str, AUTOENCODER_SHAPES)) if AUTOENCODER_SHAPES != None else ''} ${'--mask ' + ' '.join(MASK) if MASK else ''} --log_level ${LOG_LEVEL} ${'--autoencoder' if AUTOENCODER else ''} --random_restarts ${RANDOM_RESTARTS} ${' --subselect ' if SUBSELECT==True else ''} --batchifier_class ${BATCHIFIER_CLASS} --shared_entity_types ${SHARED_ENTITY_TYPES}",
                other_args=["DEPTH", "MAX_EPOCHS", "LEARNING_RATE", "RANDOM_SEED", "PATIENCE", "MOMENTUM", "BATCH_SIZE",
                            "EMBEDDING_SIZE", "HIDDEN_SIZE", "FIELD_DROPOUT", "HIDDEN_DROPOUT", "EARLY_STOP"],
                USE_GPU=env["USE_GPU"],
            )
        ),
        "ApplyModel" : env.Builder(
            **env.ActionMaker(
                "python",
                "scripts/apply_model.py",
                "--model ${SOURCES[0]} --dataset ${SOURCES[1]} ${'--split ' + SOURCES[2].rstr() if len(SOURCES) == 3 else ''} --output ${TARGETS[0]} ${'--gpu' if USE_GPU else ''}",
            )
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
    },
    tools=["default"],
)


# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)


# and the command-printing function
env['PRINT_CMD_LINE_FUNC'] = print_cmd_line


# and how we decide if a dependency is out of date
env.Decider("timestamp-newer")


def run_experiment(env, experiment_config, **args):
    data = sum([env.Glob(env.subst(p)) for p in experiment_config.get("DATA_FILES", [])], [])
    schema = experiment_config.get("SCHEMA", None)
    title = experiment_name.replace("_", " ").title().replace(" ", "")
    data = getattr(env, "preprocess_{}".format(experiment_name))("work/${EXPERIMENT_NAME}/data.json.gz",
                                                                 data, **args)
    env.Alias("data", data)

    # prepare the final spec and dataset
    observed_schema, dataset = env.PrepareDataset(["work/${EXPERIMENT_NAME}/schema.json.gz", "work/${EXPERIMENT_NAME}/dataset.pkl.gz"],
                                                  [data] + ([] if schema == None else [schema]),
                                                  **args)
    env.Alias("datasets", dataset)

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
    
    split_names = [n for n, _ in experiment_config.get("SPLIT_PROPORTIONS", env["SPLIT_PROPORTIONS"])]
    split_props = [p for _, p in experiment_config.get("SPLIT_PROPORTIONS", env["SPLIT_PROPORTIONS"])]    

    
    train, dev, test = env.SplitData(["work/${{EXPERIMENT_NAME}}/{0}_ids.txt.gz".format(n) for n in split_names], 
                                     dataset,
                                     **experiment_config,
                                     **args, RANDOM_SEED=0, PROPORTIONS=split_props)
    env.Alias("splits", [train, dev, test])
    
    # expand training configurations
    train_configs = [[]]
    for arg_name, values in experiment_config.get("TRAIN_CONFIG", {}).items():        
       train_configs = sum([[config + [(arg_name.upper(), v)] for config in train_configs] for v in values], [])
    train_configs = [dict(config) for config in train_configs]
    
    # expand apply configurations
    apply_configs = [[]]
    for arg_name, values in experiment_config.get("APPLY_CONFIG", {}).items():
       apply_configs = sum([[config + [(arg_name.upper(), v)] for config in apply_configs] for v in values], [])
    apply_configs = [dict(config) for config in apply_configs]
    
    results = []
    for config in train_configs:
        args["TRAIN_CONFIG_ID"] = md5(str(sorted(list(config.items()))).encode()).hexdigest()
        model, trace = env.TrainModel(["work/${EXPERIMENT_NAME}/model_${TRAIN_CONFIG_ID}.pkl.gz", 
                                       "work/${EXPERIMENT_NAME}/trace_${TRAIN_CONFIG_ID}.json.gz"],
                                      [dataset, train, dev],
                                      **args,
                                      **experiment_config,
                                      **config)
        env.Alias("models", model)
        for apply_config in apply_configs:
            config.update(apply_config)
            args["APPLY_CONFIG_ID"] = md5(str(sorted(list(config.items()))).encode()).hexdigest()
            output = env.ApplyModel("work/${EXPERIMENT_NAME}/${FOLD}/output_${APPLY_CONFIG_ID}.json.gz",
                                    [model, dataset],
                                    **args,
                                    **experiment_config,
                                    **config)
            env.Alias("outputs", output)

            tsne = env.MakeTSNE("work/${EXPERIMENT_NAME}/${FOLD}/tsne_${APPLY_CONFIG_ID}.json.gz",
                                [schema, output],
                                **args,
                                **experiment_config,
                                **config)
            env.Alias("tsne", tsne)
    return model


env.AddMethod(run_experiment, "RunExperiment")


#
# Run all experiments
#
models = []
names = []
for experiment_name, experiment_config in env["EXPERIMENTS"].items():
    names.append(experiment_name)
    models.append(env.RunExperiment(experiment_config, EXPERIMENT_NAME=experiment_name))
