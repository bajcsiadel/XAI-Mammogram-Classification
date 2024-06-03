import os
import pathlib

from hydra.types import RunMode
from omegaconf import omegaconf

from xai_mam.utils.environment import get_env

log_created = {}


def _get_package_name():
    import traceback

    source_location = pathlib.Path(get_env("MODULE_PATH"))

    stack = traceback.extract_stack()
    i = 0
    source_file = None
    while i < len(stack) and source_file is None:
        current_file = pathlib.Path(stack[i].filename)
        if current_file.is_relative_to(source_location):
            source_file = current_file
        i += 1

    package_location = source_file.relative_to(source_location).with_suffix("")
    return ".".join(package_location.parts)


def _get_run_location():
    return pathlib.Path(get_env("RUNS_PATH"), _get_package_name())


def _create(run_mode, sweep_dir, filename="progress.log"):
    if run_mode == RunMode.RUN:
        # do not create progress file
        return os.devnull
    if type(sweep_dir) is not pathlib.Path:
        sweep_dir = pathlib.Path(sweep_dir)

    filename = sweep_dir / filename

    global log_created

    if filename not in log_created or not log_created[filename]:
        log_created[filename] = True
        filename.parent.mkdir(parents=True, exist_ok=True)
    return filename


def resolve_is_backbone_only():
    def is_backbone_only(backbone_only):
        return "only-" if backbone_only else ""

    omegaconf.OmegaConf.register_new_resolver("is_backbone_only", is_backbone_only)


def resolve_is_debug_mode():
    def is_debug_mode(debug):
        return "debug-" if debug else ""

    omegaconf.OmegaConf.register_new_resolver("is_debug_mode", is_debug_mode)


def resolve_model_type():
    def model_type(backbone_only):
        return "backbone" if backbone_only is not None else "explainable"

    omegaconf.OmegaConf.register_new_resolver("model_type", model_type)


def resolve_run_location():
    omegaconf.OmegaConf.register_new_resolver("run_location", _get_run_location)


def resolve_package_name():
    omegaconf.OmegaConf.register_new_resolver("package_name", _get_package_name)


def resolve_create():
    omegaconf.OmegaConf.register_new_resolver("create", _create)


def resolve_override_dirname():
    def sanitize_override_dirname(override_dirname):
        override_dirname = override_dirname.replace("model.phases", "")
        override_dirname = override_dirname.replace("model.params", "")
        override_dirname = override_dirname.replace("learning_rate", "LR")
        override_dirname = override_dirname.replace("epochs", "E")
        return override_dirname.replace("/", "-")

    omegaconf.OmegaConf.register_new_resolver(
        "sanitize_override_dirname", sanitize_override_dirname
    )


def add_all_custom_resolvers():
    resolve_create()
    resolve_is_backbone_only()
    resolve_is_debug_mode()
    resolve_model_type()
    resolve_override_dirname()
    resolve_package_name()
    resolve_run_location()
