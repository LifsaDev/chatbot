import json

mylist = [("hello", "hi"), ("he", "ha")]

# Dump the list of tuples to a JSON file
with open("mylist.json", "w") as file:
    json.dump(mylist, file)

# Load the list of lists from the JSON file
loaded_list = []
with open("mylist.json", "r") as file:
    loaded_list = json.load(file)

# Convert the loaded list of lists to a list of tuples


print(loaded_list)
