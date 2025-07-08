from __future__ import absolute_import
import argparse
import logging
import logging.config
from benchmark.datasets import DATASETS
from benchmark.algorithms.definitions import (get_specific_algorithm_definitions)
import glob
import os
import psutil
import json

from benchmark.algorithms.definitions import (Definition,
                                               instantiate_algorithm)
from benchmark.algorithms.streaming_runner import StreamingRunner
from benchmark.streaming.load_runbook import load_runbook
from benchmark.results import store_results

def run(definition, dataset, count, run_count, 
        runbook_path="neurips23/streaming/simple_runbook.yaml"):
    algo = instantiate_algorithm(definition)
    assert not definition.query_argument_groups \
           or hasattr(algo, "set_query_arguments"), """\
            error: query argument groups have been specified for %s.%s(%s), but the \
            algorithm instantiated from it does not implement the set_query_arguments \
            function""" % (definition.module, definition.constructor, definition.arguments)

    ds = DATASETS[dataset]()

    distance = ds.distance()
    search_type = ds.search_type()
    build_time = -1 # default value used to indicate that the index was loaded from file
    print(f"Running {definition.algorithm} on {dataset}")
    
    streaming_runner = StreamingRunner
    max_pts, runbook = load_runbook(dataset, ds.nb, runbook_path)

    try:
        # Try loading the index from the file
        memory_usage_before = algo.get_memory_usage()
        streaming_runner.build(algo, dataset, max_pts)

        index_size = algo.get_memory_usage() - memory_usage_before
        print('Index memory footprint: ', index_size)

        print("Starting query")
        query_argument_groups = definition.query_argument_groups

        # Make sure that algorithms with no query argument groups still get run
        # once by providing them with a single, empty, harmless group
        if not query_argument_groups:
            query_argument_groups = [[]]
        for pos, query_arguments in enumerate(query_argument_groups, 1):
            print("Running query argument group %d of %d..." %
                    (pos, len(query_argument_groups)))
            if query_arguments:
                algo.set_query_arguments(query_arguments)
            descriptor, results = streaming_runner.run_task(
                    algo, ds, distance, count, 1, search_type, False, runbook)
            descriptor["build_time"] = build_time
            descriptor["index_size"] = index_size
            descriptor["algo"] = definition.algorithm
            descriptor["dataset"] = dataset
            print('start store results')
            store_results(dataset, count, definition,
                            query_arguments, descriptor,
                            results, search_type, runbook_path)
            print('end store results')
    finally:
        algo.done()

def run_from_cmdline(args=None):
    parser = argparse.ArgumentParser('''

            NOTICE: You probably want to run.py rather than this script.

    ''')
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        help=f'Dataset to benchmark on.',
        required=True)
    parser.add_argument(
        '--algorithm',
        help='Name of algorithm for saving the results.',
        required=True)
    parser.add_argument(
        '--module',
        help='Python module containing algorithm. E.g. "ann_benchmarks.algorithms.annoy"',
        required=True)
    parser.add_argument(
        '--constructor',
        help='Constructer to load from module. E.g. "Annoy"',
        required=True)
    parser.add_argument(
        '--count',
        help='k: Number of nearest neighbours for the algorithm to return.',
        required=True,
        type=int)
    parser.add_argument(
        '--runs',
        help='Number of times to run the algorihm. Will use the fastest run-time over the bunch.',
        required=True,
        type=int)
    parser.add_argument(
        'build',
        help='JSON of arguments to pass to the constructor. E.g. ["angular", 100]'
        )
    parser.add_argument(
        'queries',
        help='JSON of arguments to pass to the queries. E.g. [100]',
        nargs='*',
        default=[])
    parser.add_argument(
        '--power-capture',
        help='Power capture parameters for the T3 competition. '
            'Format is "ip:port:capture_time_in_seconds (ie, 127.0.0.1:3000:10).',
        default="")
    parser.add_argument(
        '--runbook_path',
        help='runbook yaml path for neurips23 streaming track',
        default='neurips23/streaming/simple_runbook.yaml'
    )

    args = parser.parse_args(args)
    algo_args = json.loads(args.build)
    query_args = [json.loads(q) for q in args.queries]

    # TODO:
    # if args.power_capture:
    #     power_capture( args.power_capture )
    #     power_capture.ping()

    definition = Definition(
        algorithm=args.algorithm,
        docker_tag=None,  # not needed
        docker_volumes=[],
        module=args.module,
        constructor=args.constructor,
        arguments=algo_args,
        query_argument_groups=query_args,
        disabled=False
    )
    run(definition, args.dataset, args.count, args.runs, args.runbook_path)

def run_no_docker(definition, dataset, count, runs, 
                  cpu_limit, mem_limit=None, power_capture=None,
                  runbook_path='neurips23/streaming/simple_runbook.yaml'):
    cmd = ['--dataset', dataset,
           '--algorithm', definition.algorithm,
           '--module', definition.module,
           '--constructor', definition.constructor,
           '--runs', str(runs),
           '--count', str(count)]
     
    cmd += ["--runbook_path", runbook_path]

    cmd.append(json.dumps(definition.arguments))
    cmd += [json.dumps(qag) for qag in definition.query_argument_groups]
    run_from_cmdline(cmd)