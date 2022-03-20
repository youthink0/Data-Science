import sys
from collections import defaultdict
from itertools import chain, combinations
import time

'''
###---Reference---
About FP-growth Algorithm
    method: https://www.itread01.com/content/1546888205.html
    implement: https://towardsdatascience.com/fp-growth-frequent-pattern-generation-in-data-mining-with-python-implementation-244e561ab1c3
###------
'''

###---Define Node---
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
###------


###---each Set Get Frequency and Output---
def getSupport(testSet, itemSetList):
    count = 0
    for itemSet in itemSetList:
        if(set(testSet).issubset(itemSet)):
            count += 1
    return count

def SetFrequency(freqItemSet, itemSetList, inputLen,  output_name):
    rules = []
    ResStdout = ""
    original_stdout = sys.stdout # Save a reference to the original standard output
    #Open OutputFile
    with open(output_name, 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        for itemSet in freqItemSet:
            #get Set Frequency
            itemSetSup = getSupport(itemSet, itemSetList)
            SetResult = ",".join(map(str,itemSet))
            SetSupport = round(itemSetSup/inputLen, 4)
            
            #OutputFormat
            ResStdout = SetResult + ":" + "%.4f"%SetSupport
            print(ResStdout)
    sys.stdout = original_stdout # Reset the standard output to its original value
    return 
###------

###---MineTree---
def mineTree(headerTable, minSup, preFix, freqItemList):
    # Sort the items with frequency, AKA p[1][0], and create a list
    sortedItemList = [item[0] for item in sorted(list(headerTable.items()), key=lambda p:p[1][0])] 
    #print("item: ", sortedItemList)
    #ex : ['3', '6', '5', '4', '2', '1', '8', '7', '10', '9', '0'] freq:low->high
    sortedFreqList = [item[1] for item in sorted(list(headerTable.items()), key=lambda p:p[1][0])]
    #print("freq: ", sortedFreqList)

    for item in sortedItemList:  
        # Pattern growth is achieved by the concatenation of suffix pattern with frequent patterns generated from conditional FP-tree
        newFreqSet = preFix.copy()
        newFreqSet.add(item)
        freqItemList.append(newFreqSet)
        # Find all prefix path, constrcut conditional pattern base
        conditionalPattBase, frequency = findPrefixPath(item, headerTable) 
        # Construct conditonal FP Tree with conditional pattern base
        conditionalTree, newHeaderTable = constructTree(conditionalPattBase, frequency, minSup) 
        if newHeaderTable != None:
            # Mining recursively on the tree
            mineTree(newHeaderTable, minSup,
                       newFreqSet, freqItemList)

def ascendFPtree(node, prefixPath):
    if node.parent != None:
        prefixPath.append(node.itemName)
        ascendFPtree(node.parent, prefixPath)

def findPrefixPath(basePat, headerTable):
    # First node in linked list
    treeNode = headerTable[basePat][1] 
    condPats = []
    frequency = []
    while treeNode != None:
        prefixPath = []
        # From leaf node all the way to root
        ascendFPtree(treeNode, prefixPath)  
        if len(prefixPath) > 1:
            # Storing the prefix path and it's corresponding count
            condPats.append(prefixPath[1:])
            frequency.append(treeNode.count)

        # Go to next node
        treeNode = treeNode.next  
    return condPats, frequency
###------


###---getConstructTree---
def constructTree(itemSetList, frequency, minSup):
    headerTable = defaultdict(int)
    # Counting frequency and create header table
    for idx, itemSet in enumerate(itemSetList):
        for item in itemSet:
            headerTable[item] += frequency[idx]
    #print(headerTable)
    #headerTable ex = {'5': 6, '9': 11, '10': 10, '0': 13, '1': 7, '4': 6, '6': 5, '8': 7, '3': 4, '2': 6, '7': 7}

    # Deleting items below minSup
    headerTable = dict((item, sup) for item, sup in headerTable.items() if sup >= float(minSup))
    # If nothing can fit munSup
    if(len(headerTable) == 0):
        return None, None

    # HeaderTable column [Item: [frequency, headNode]]
    for item in headerTable:
        headerTable[item] = [headerTable[item], None]
    #headerTable ex : {'8585': [1, None], '948': [1, None], '1125': [1, None]}    

    # Init Null head node
    fpTree = Node('Null', 1, None)
    # Update FP tree for each cleaned and sorted itemSet
    for idx, itemSet in enumerate(itemSetList):
        itemSet = [item for item in itemSet if item in headerTable]
        #descend sorted by Frequency, AKA headerTable[item][0]
        itemSet.sort(key=lambda item: headerTable[item][0], reverse=True)
        # Traverse from root to leaf, update tree with given item
        currentNode = fpTree
        for item in itemSet:
            currentNode = updateTree(item, currentNode, headerTable, frequency[idx])

    return fpTree, headerTable
 
def updateHeaderTable(item, targetNode, headerTable):
    if(headerTable[item][1] == None):
        headerTable[item][1] = targetNode
    else:
        currentNode = headerTable[item][1]
        # Traverse to the last node then link it to the target
        while currentNode.next != None:
            currentNode = currentNode.next
        currentNode.next = targetNode

def updateTree(item, treeNode, headerTable, frequency):
    if item in treeNode.children:
        # If the item already exists, increment the count
        treeNode.children[item].increment(frequency)
    else:
        # Create a new branch
        newItemNode = Node(item, frequency, treeNode)
        treeNode.children[item] = newItemNode
        # Link the new branch to header table
        updateHeaderTable(item, newItemNode, headerTable)

    return treeNode.children[item]
###------


###---Data Loading--- 
def load_data(input_name):
    itemSetList = []
    frequency = []

    with open(input_name, 'r') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            line_data = line.split(",")
            #print(line_data)
            # ex : itemSetList.append(line)
            itemSetList.append(line_data)
            frequency.append(1)
    return itemSetList, frequency

def extract_value(line):
    redundant = [',', '\n']
    return False if line in redundant else True
###------

if __name__== "__main__" :    
    time_start = time.time() #開始計時
    ###---Augment Reading---
    minsup_ratio =  sys.argv[1] #minsupport
    input_name = sys.argv[2] #input file
    output_name = sys.argv[3] #output_file
    
    ###---Data Loading---
    itemSetList, frequency = load_data(input_name)
    #print(itemSetList, frequency)
    #itemSetList ex = [['112', '1034'], ['647', '704', '929']]    

    ###---getConstructTree---
    inputLen = len(itemSetList)
    minSup = inputLen * float(minsup_ratio)
    print("minSup:", minSup, "Testcase:",inputLen, "input_minsup:",minsup_ratio)
    fpTree, headerTable = constructTree(itemSetList, frequency, minSup)
    #print(fpTree, headerTable)
    #ex : <__main__.Node object at 0x7f6ab32ff040> {'5': [6, <__main__.Node object at 0x7f6ab1f1e3a0>], '9': [11, <__main__.Node object at 0x7f6ab326eb50>]    

    if(fpTree == None):
        print('No frequent item set')
    else:
        freqItems = []
        #list can return what they append in function
        mineTree(headerTable, minSup, set(), freqItems)
        SetFrequency(freqItems, itemSetList, inputLen, output_name)
        #print(freqItems)
    
    time_end = time.time()    #結束計時
    time_c= time_end - time_start   #執行所花時間
    print('time cost', time_c, 's')
    
