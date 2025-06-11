from app.storage.object_reader import ObjectReader


class FileObjectReader(ObjectReader):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def read(self):
        return open(self.filepath, "rb")
