from dataclasses import fields


def filter_dict_for_dataclass(cls, params):
    field_names = {f.name for f in fields(cls)}
    return {k: v for k, v in params.items() if k in field_names}
