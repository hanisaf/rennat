import utils
class index:
    def __init__(self, index_file: str = None) -> None:
        if index_file:
            self.index = utils.load_store(index_file)
        pass