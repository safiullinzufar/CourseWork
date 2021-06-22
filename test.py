import json

with open("config.json", "r") as read_file:
    data = json.load(read_file)
print(data['settings']['n_classes'])
