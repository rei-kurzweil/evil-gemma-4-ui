import importlib.util
import os


def load_perspective_parser():
    module_path = os.path.join(os.path.dirname(__file__), "perspective-parser.py")
    spec = importlib.util.spec_from_file_location("perspective_parser", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PerspectiveParser


PerspectiveParser = load_perspective_parser()


class Demultiplexer:
    def __init__(self):
        self.parser = PerspectiveParser()

    def route_sentence(self, sentence: str):
        self.parser.feed_sentence(sentence)

    def pretty_print(self):
        return self.parser.pretty_print()
