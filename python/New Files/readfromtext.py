from testCaseModifier import GCheckModifier
from SQLChecker import parser
file1 = open('vals.txt', 'r')
count=1
Lines = file1.readlines()
arr=[]
for line in Lines:
    line=line.strip()
    arr.append(line)
    #print(line)
g=GCheckModifier()
p=parser()
count=0
for i in range(len(arr)):
    val=g.grammarchecker(arr[i])
    k=p.main(val)
    if k==0:
        #print("pass")
        #print(val)
        #print("-------------------------------------")
        #count=count+1
        continue
    else:
        print("Failed")
        print(val)
        print("-------------------------------------")
        count=count+1
print(count)