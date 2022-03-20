from collections import defaultdict
from csv import reader

''' 
###----defaultdict test----
s = [('red', 1), ('blue', 2), ('red', 3), ('blue', 4), ('red', 1), ('blue', 4)]
d = defaultdict(set)
for k, v in s:
    d[k].add(v)
print(d)

class Node:
    def __init__(self, itemName, frequency, parentNode):
        self.itemName = itemName
        self.count = frequency
        self.parent = parentNode
        self.children = {}
        self.next = None

    def increment(self, frequency):
        self.count += frequency

    def display(self, ind=1):
        print('  ' * ind, self.itemName, ' ', self.count)
        for child in list(self.children.values()):
            child.display(ind+1)

###----load data test---
def getFromFile(fname):
    itemSetList = []
    frequency = []
    
    with open(fname, 'r') as file:
        csv_reader = reader(file)
        for line in csv_reader:
            line = list(filter(None, line))
            itemSetList.append(line)
            frequency.append(1)

    return itemSetList, frequency


def constructTree(itemSetList, frequency, minSup):
    headerTable = defaultdict(int)
    # Counting frequency and create header table
    for idx, itemSet in enumerate(itemSetList):
        for item in itemSet:
            headerTable[item] += frequency[idx]
    #headerTable ex = {'8585': 1, '1289': 1, '1544': 1, '1651': 1, '1250': 2}

    # Deleting items below minSup
    headerTable = dict((item, sup) for item, sup in headerTable.items() if sup >= minSup)
    # If nothing can fit munSup
    if(len(headerTable) == 0):
        return None, None

    # HeaderTable column [Item: [frequency, headNode]]
    for item in headerTable:
        headerTable[item] = [headerTable[item], None]
    print(headerTable)
    #headerTable ex : {'8585': [1, None], '948': [1, None], '1125': [1, None]}    

     # Init Null head node
    fpTree = Node('Null', 1, None)
    print(fpTree)
    
    # Update FP tree for each cleaned and sorted itemSet
    for idx, itemSet in enumerate(itemSetList):
        itemSet = [item for item in itemSet if item in headerTable]
        itemSet.sort(key=lambda item: headerTable[item][0], reverse=True)
        print(itemSet)
        # Traverse from root to leaf, update tree with given item
        currentNode = fpTree
        
        for item in itemSet:
            currentNode = updateTree(item, currentNode, headerTable, frequency[idx])
    
    return fpTree, headerTable

if __name__== "__main__" :
    itemSetList, frequency=getFromFile("data.csv")
    #print(itemSetList, frequency)

    fpTree, headerTable = constructTree(itemSetList, frequency, 1)
'''
def freq_test(freqi):
    freqi=2
    return
freqitem=1
freq_test(freqitem)
print(freqitem)

