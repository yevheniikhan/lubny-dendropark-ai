import numpy as np, re, sys
A='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
def dec(body,w,h):
    rows=body.split('/')
    out=[]
    for r in rows:
        if r=='' : continue
        v=[];i=0
        while i<len(r):
            c=r[i]
            if c=='.':
                m=re.match(r'\.(\d+);',r[i:])
                if m: v+= [np.nan]*int(m.group(1)); i+=m.end()
                else: v.append(np.nan); i+=1
            else: v.append(A.index(c)); i+=1
        out.append(v)
    return out
txt=open(sys.argv[1]).read()
res={}
for line in txt.strip().split('\n'):
    name,w,h,body=line.split(' ',3)
    rows=dec(body,int(w),int(h))
    lens=set(len(r) for r in rows)
    print(name,w,h,len(rows),lens)
    res[name]=np.array(rows,float)
np.savez(sys.argv[2],**res)
