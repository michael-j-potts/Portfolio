import numpy as np

def GraphMatrix(GraphConnections):
    #Takes GraphConnections and converts it into a connection matrix of 0s and 1s
    count = 0
    NameList = []
    GraphList = []
    masklist = []
    for Node in GraphConnections:
        NodeList = []
        for connections in Node.strip(' ').split(' '):
            if count == 0:
                NameList.append(connections)
            else:
                if (connections != ''):
                    NodeList.append(str(connections))
            count += 1
        GraphList.append(NodeList)    
        count = 0
    #Create mask for each document and list of masks
    for i in GraphList:
        mask = np.isin(NameList, i)
        masklist.append(mask)
    return masklist

def TransitionMatrix(GraphConnections, n):
    #Creates the transition matrix for the directed graph network
    masklist = GraphMatrix(GraphConnections)
    #Select each doc
    count = 0
    TransMatrix = []
    for doc in masklist:
        #Count trues for each doc
        DocTransition = np.zeros(n)
        URLpresent = 0
        for value in doc:
            if value == True:
                URLpresent += 1
        #To avoid dangling nodes, we can set nodes with no outgoing url 
        #links, to have an equally possible chance to travel to any node
        #so we will set each link to True. This avoids dividing by 0
        #and allows for the pagerank algorithm to sum to 1.
        if URLpresent == 0:
            iter = 0
            for iter in range(len(doc)):
                doc[iter] = True
                iter += 1
                URLpresent += 1
        #convert each true value to 1/n respective to each doc
        for count in range(len(doc)):
            if doc[count] == True:
                DocTransition[count] = 1 / URLpresent
        count += 1
        TransMatrix.append(DocTransition)
    return np.array(TransMatrix)

def PageRank(TransMatrix, alpha, n, max_iterations, epsilon):
    #Calculate the page rank algorithms for the given wikipedia documents.
    print("Running PageRank.")
    Mupdate = alpha * TransMatrix.T
    iupdate = (1 - alpha) / n
    a = Mupdate + iupdate
    InitialPvector = np.full((n), 1/n, dtype = float)
    iter = 0
    for iter in range(max_iterations):
        FinalPvector = a @ InitialPvector
        #print(sum(abs(FinalPvector - InitialPvector)))
        #print("Epsilon: ", epsilon)
        if sum(abs(FinalPvector - InitialPvector)) < epsilon:
            break
        InitialPvector = FinalPvector
    print("Converged after ", iter, " iteration(s).")
    #print(FinalPvector)
    return FinalPvector 

#Set our directory, open necessary files, and import and sum number of nodes.
np.set_printoptions(threshold = np.inf)
Directory = './Repository/'
GraphConnections = open((Directory + "Trimmed.txt"), 'r').read().split('\n')
alpha = 0.2
n = 0
max_iterations = 50
epsilon = 0.0001
names = open((Directory + 'Names.txt'), 'r').readlines()
for i in open((Directory + 'metadata.txt'), 'r').read().split('\n'):
    if i != '':
        n += int(i)    

TransMatrix = TransitionMatrix(GraphConnections, n)
Ranks = np.reshape(PageRank(TransMatrix, alpha, n, max_iterations, epsilon), (-1,1))
namelist = []
for i in names:
    namelist.append(i.strip())
namelist = np.reshape(np.array(namelist), (-1, 1))

NameRanks = np.concatenate((namelist, Ranks), axis = 1)
NameRanks = NameRanks[NameRanks[:,1].argsort()][::-1]

print("The 5 documents with the highest page rank score are: ")
for i in range(5):
    print(i+1, NameRanks[i][0], np.round(float(NameRanks[i][1]), 5))
print()
print("The 5 documents with the lowest page rank score are: ")
for i in range(5):
    print(i+1, NameRanks[-1 * i-1][0], np.round(float(NameRanks[-1 * i-1][1]), 5))
