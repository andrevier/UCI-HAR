"""Notes on important code snippets in the file."""
import numpy as np

listOfNumbers = []
txt = "5. 5.1 2.2 2.6 1.1\n7.1 6.2 9.2 5.4 8.3\n1.2 1.1 1.7 2.4 1.5"
txt1 = "1. 2.1 4.2 6.6 7.1\n1.1 3.2 4.2 5.4 3.3\n4.2 5.1 9.7 4.4 5.5"

data1 = txt.splitlines()
data1 = map(lambda y: y.rstrip().lstrip().split(), data1)
data1 = [list(map(float, line)) for line in data1]
listOfNumbers.append(data1)

data2 = txt1.splitlines()
data2 = map(lambda y: y.rstrip().lstrip().split(), data2)
data2 = [list(map(float, line)) for line in data2]
listOfNumbers.append(data2)

arrayOfNumbers = np.array(listOfNumbers)

print("dim: ", arrayOfNumbers.shape)
print(arrayOfNumbers)
print("........")
arrayOfNumbers = np.transpose(arrayOfNumbers, (1, 2, 0))
print("dim: ", arrayOfNumbers.shape)
print(arrayOfNumbers)
