from PIL import Image; import numpy as np
from pyproj import Transformer
from scipy.ndimage import map_coordinates
M=Image.new('RGB',(1400,1440))
for i in range(6):
    r,c=divmod(i,2); M.paste(Image.open('esri/m%d.jpg'%i).crop((81,63,781,543)),(c*700,r*480))
M.save('esri/mosaic_3857.jpg',quality=92); M=np.array(M).astype(float)
MX0=3671855;MY1=6444190;res=0.8
X0,X1,Y0,Y1=498920,499600,5536580,5537280
r=0.5;W=int((X1-X0)/r);H=int((Y1-Y0)/r)
X,Y=np.meshgrid(X0+(np.arange(W)+0.5)*r, Y1-(np.arange(H)+0.5)*r)
mx,my=Transformer.from_crs(32636,3857,always_xy=True).transform(X,Y)
cc=(mx-MX0)/res-0.5; rr=(MY1-my)/res-0.5
out=np.stack([map_coordinates(M[...,k],[rr,cc],order=1,mode='nearest') for k in range(3)],-1)
Image.fromarray(np.clip(out,0,255).astype(np.uint8)).save('esri/ortho_utm.png')
print(cc.min(),cc.max(),rr.min(),rr.max())
