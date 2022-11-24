import pandas as pd
import numpy as np

def func23():
   
    l= [1,34,5,100,2]
    l.sort(reverse=True)
    return l

def listar():
    xx=[1,2,3,4,5,6,7,8,9,10,11]
    return (list(filter(lambda x: x %2 ==0,xx)))

def colors():
    lista=[22,33,44,55,66]
    g=int(np.average(lista))
    print(g)
    
def arregloletras(str):
    lista=[]
    for i in str:
        lista.append(i.upper())
    print(''.join(lista))

def palindrome(str_):
    str_rev= str_[::-1]
    return str_ == str_rev

def mirrow(str_2):
    count=0
    half= int( len(str_2)/2)
    str_rev=str_2[::-1]
    
    for i in range(len(str_2)-2):
        if( str_2[i] != str_rev[i]):
            count+=1
    if count<=2:
        return True
    else:
        return False

def matriz():
    la= [[1,2,3],[4,5,6], [7,8,9]]
    
    for i in range(len(la)):
        for j in range(len(la[0])):
            print(la[i][j])
    
matriz()    
#colors()


def tri():
    languages = ['Python', 'C', 'C++', 'C#', 'Java']
    for i,language in enumerate(languages):
        print(i,language)
        i+=1

tri()

dataframe={ "val":[1,2,3,4],
            "nombre":["al,g)","el,g","il,g","ol,g"]
    }

l= pd.DataFrame(dataframe)
df=pd.DataFrame(dataframe)
l=l['nombre'].str.strip('ñ')

df['df']= df['nombre'].str.strip(')')

df[['uno','dos']]= df.nombre.str.split("," ,expand=True)

x = df['val'].agg
df[['First','Last']] = df.nombre.str.split(",", expand=True)
answer= df[['val']].agg('max')
#print("este es",answer)
#print(df)
str="anitalavalatina"
str_2="anitalavalatina"
#print(func23())
#print(listar())
#arregloletras(str)
print(mirrow(str_2))
