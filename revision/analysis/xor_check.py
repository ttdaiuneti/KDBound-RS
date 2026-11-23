"""R2.5: XOR table. Pawlak dependency gamma, Liang 2013 complementary conditional entropy E(D|B), NRS gamma (delta=0.5)."""
from itertools import product
U=[(a,b,a^b) for a,b in product((0,1),repeat=2)]
def blocks(B): 
    g={}
    for i,x in enumerate(U): g.setdefault(tuple(x[j] for j in B),[]).append(i)
    return list(g.values())
def gamma(B): return sum(len(X) for X in blocks(B) if len({U[i][2] for i in X})==1)/len(U)
def E(B):
    n=len(U); Y=[[i for i in range(n) if U[i][2]==d] for d in (0,1)]
    return sum(len(set(X)&set(y))/n*len(set(X)-set(y))/n for X in blocks(B) for y in Y)
def gnrs(B,delta=0.5):
    pos=0
    for x in U:
        ne=min(sum((x[j]-y[j])**2 for j in B)**.5 for y in U if y[2]!=x[2]); pos+= ne>delta
    return pos/len(U)
for name,f in (("Pawlak gamma",gamma),("NRS gamma d=0.5",gnrs)):
    print(name,"sig(a1,{})=",f([0])-f([]),"sig(a1,{a2})=",f([0,1])-f([1]))
print("Sig3(a1,{}) = E(D|{})-E(D|{a1}) =",E([])-E([0]))
