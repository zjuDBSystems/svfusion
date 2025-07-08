import os
import yaml

def docker_tag_base():
    return 'neurips23'

def basedir():
    return 'neurips23'

def docker_tag(track, algo):
    return docker_tag_base() + '-' + track + '-' + algo

def dockerfile_path_base():
    return os.path.join('neurips23', 'Dockerfile')

def track_path(baseline_method):
    return os.path.join('baselines', baseline_method)

def dockerfile_path(track, algo):
    return os.path.join(track_path(track), algo, 'Dockerfile')

def yaml_path(track, algo):
    return os.path.join(track_path(track), algo, 'config.yaml')

def get_definitions(track, algo):
    return yaml.load(yaml_path(track, algo))



