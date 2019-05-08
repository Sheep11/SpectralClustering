import numpy as np

def load_data(path):
    data=[]
    with open(path) as f:
        for line in f.readlines():
            if line=="\n":
                break

            feature=line.split(',')
            feature.pop()
            data.append(list(map(float,feature)))

    return np.array(data)

def accuracy(res):
    res=list(res)

    acc=[0,0,0]
    tmp=[0,0,0]

    for i in range(3):
        tmp[i]=res[:50].count(i)
    acc[0]=max(tmp)

    for i in range(3):
        tmp[i]=res[50:100].count(i)
    acc[1]=max(tmp)

    for i in range(3):
        tmp[i]=res[100:150].count(i)
    acc[2]=max(tmp)

    print('accuracy: ' + str(sum(acc)/150))
    return sum(acc)/150