import json

idx2char = []

# Creating a mapping from unique characters to indices
with open("char2idx.json", "r") as json_file:
    char2idx = json.load(json_file)
    json_file.close()

arr_file = open("idx2char.txt", "r")
while 1: 
      
    # read by character 
    char = arr_file.read(1)           
    if not char: 
        break
    idx2char.append(char)
    
arr_file.close() 

print(idx2char)

