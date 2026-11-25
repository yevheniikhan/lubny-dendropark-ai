import glob, zlib, sys
A='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
ch={}
for f in sorted(glob.glob('/home/claude/lubny/data/raw2/z*.txt')):
    for line in open(f):
        if line.strip():
            i,rest=line.strip().split(' ',1); ch[int(i)]=rest.rsplit(' #',1)[0].replace(' ','')
s=''.join(ch[i] for i in range(len(ch)))
n_bytes=int(sys.argv[1])
out=bytearray()
for k in range(0,len(s),11):
    n=0
    for c in s[k:k+11]: n=n*62+A.index(c)
    out+=n.to_bytes(8,'big')
out=bytes(out[:n_bytes])
txt=zlib.decompress(out,-15).decode()
open('/home/claude/lubny/data/park2_txt.txt','w').write(txt)
print(len(s),len(txt),[l[:12] for l in txt.split('\n')])
