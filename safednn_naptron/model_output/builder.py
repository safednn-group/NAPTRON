from .base import IdentityHandler
from .simple_dump import SimpleHandler

OUTPUT_HANDLERS = {
    "identity": IdentityHandler,
    "simple_dump": SimpleHandler
}


def build_output_handler(cfg):
    """Build output handler."""
    obj_cls = OUTPUT_HANDLERS.get("identity")
    if cfg.get("output_handler") is None:
        return obj_cls()

    obj_type = cfg.output_handler.get('type')
    if isinstance(obj_type, str):
        if obj_type in OUTPUT_HANDLERS:
            obj_cls = OUTPUT_HANDLERS.get(obj_type)
        else:
            raise Warning("Specified output handler is not registered; identity will be used.")

    return obj_cls(cfg)
