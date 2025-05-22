import json


class Tracker(dict):
    def __init__(self):
        self.update({"epoch": 0, "train": dict(), "valid": dict()})

    def print(self, section: str="train") -> str:
        return json.dumps({section+"/"+k: v for k,v in self[section].items()})