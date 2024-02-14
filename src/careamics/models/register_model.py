from torch.nn import Module


CUSTOM_MODELS = {}


def register_model(name):
    
    if name in CUSTOM_MODELS:
        raise ValueError(
            f"Model {name} already exists."
        )
        
    def add_custom_model(model: Module) -> Module:

        CUSTOM_MODELS[name] = model

        return model

    return add_custom_model


def get_custom_model(name: str) -> Module:

    if not name in CUSTOM_MODELS:
        raise ValueError(
            f"Model {name} is unknown."
        )

    return CUSTOM_MODELS[name]
        