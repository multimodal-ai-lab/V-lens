"""The unified models, loaded by name: load("bagel"), load("show_o2"), load("sensenova"), load("hunyuan"),
load("liquid"), load("janus_pro"). Each model's code needs its own environment (see README.md, requirements/)."""

MODELS = ("bagel", "show_o2", "sensenova", "hunyuan", "liquid", "janus_pro")


def load(name: str):
    if name == "bagel":
        from vlens.models.bagel import BagelModel
        return BagelModel()
    if name == "show_o2":
        from vlens.models.show_o2 import ShowO2Model
        return ShowO2Model("7B")
    if name == "sensenova":
        from vlens.models.sensenova import SenseNovaModel
        return SenseNovaModel("U1")
    if name == "janus_pro":  # Janus-Pro-7B, draws autoregressively (576 VQ codes)
        from vlens.models.janus_pro import JanusProModel
        return JanusProModel()
    if name == "liquid":  # Liquid-7B, draws autoregressively (Gemma-7B + Chameleon VQGAN codes)
        from vlens.models.liquid import LiquidModel
        return LiquidModel()
    if name == "hunyuan":  # HunyuanImage-3.0-Instruct, 80B mixture of experts, several GPUs (device_map="auto")
        from vlens.models.hunyuan import HunyuanImageModel
        return HunyuanImageModel()
    raise ValueError(f"unknown model {name!r}; known: {MODELS}")


def no_system_message(m):
    """Draw from the request alone, without the model's fixed generation system message: SenseNova-U1 only (the other
    models have none). The empty prompt used for guidance stays upstream's (it has no system message either). Not used
    for the paper's numbers (SenseNova-U1 draws from the bare request "thing" under its system message); scripts expose
    it as --no-system."""
    if not hasattr(m, "system_message"):
        raise ValueError(f"{type(m).__name__} has no fixed drawing system message to switch off")
    m.system_message = False
    return m


# The insertion layer R of each model: copying a word's hidden state into the slot "thing" of "a photo of a thing" at
# this layer makes the model draw the word (scripts/calibrate_insertion_layer.py). The V-lens carries every hidden
# state to this layer. HunyuanImage (9), Liquid (6) and Janus-Pro (8) store theirs in their lens statistics.
READING_LAYER = {"bagel": 8, "show_o2": 27, "sensenova": 24}


def transport_name(model: str, n_passages: int = 64) -> str:
    """Where the fitted layer statistics and Jacobian maps of a model are stored (vlens/transport.py)."""
    return f"{model}_transport_to{READING_LAYER[model]}_n{n_passages}"
