from model import small_cnn
from model.registry import register_model

register_model("small_cnn", small_cnn.create_model)