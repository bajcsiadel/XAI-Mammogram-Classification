import pathlib

from omegaconf import omegaconf

from xai_mam.utils.environment import get_env


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


def resolve_format_backbone_only():
    def format_backbone_only(backbone_only):
        return "only-" if backbone_only else ""

    omegaconf.OmegaConf.register_new_resolver(
        "format_backbone_only", format_backbone_only
    )


def resolve_model_type():
    def model_type(backbone_only):
        return "backbone" if backbone_only is not None else "explainable"

    omegaconf.OmegaConf.register_new_resolver("model_type", model_type)


def resolve_run_location():
    omegaconf.OmegaConf.register_new_resolver("run_location", _get_run_location)


def resolve_package_name():
    omegaconf.OmegaConf.register_new_resolver("package_name", _get_package_name)


def add_all_custom_resolvers():
    resolve_format_backbone_only()
    resolve_model_type()
    resolve_run_location()
    resolve_package_name()
