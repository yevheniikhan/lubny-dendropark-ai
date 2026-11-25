import sys, glob
ok={}
for f in sorted(glob.glob('/home/claude/lubny/data/'+__import__('os').environ.get('RAW','raw2')+'/z*.txt')):
    for line in open(f):
        line=line.strip()
        if not line: continue
        i,rest=line.split(' ',1); body,h=rest.rsplit(' #',1)
        s=body.replace(' ','')
        hh=0
        for ch in s: hh=(hh*31+ord(ch))%1000003
        ok[int(i)]=(s, hh==int(h))
print({k:v[1] for k,v in sorted(ok.items())})
