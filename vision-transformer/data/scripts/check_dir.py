import os

directory = "vision-transformer/data/cifar-10"

if os.path.exists(directory):
    print("Exists")
else:
    print("Doesnt exist")