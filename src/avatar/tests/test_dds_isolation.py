"""Adopted from root's independent probe W/avatar-root-atlas-dds-probe-001/VERIFY-AS-RUN.py (SHA-256 recorded in
tests/results/test_dds_isolation.json); changes: output under this module's work/, per-part `isolated_through` added,
summary copied to tests/results/test_dds_isolation.json. Usage: python tests/test_dds_isolation.py <new-output-name>

Read actual built PC half-UV/DXT1 data; perturb neighboring charts, including a no-gutter control.

Uses the preserved C234 read-only forest parser, not the Avatar bake/compose modules. No game/DLL loading.
An optional output name allows a later genuinely changed candidate to be checked without overwriting receipts.
"""
import hashlib
import importlib.util
import io
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import PIL
from PIL import Image

M=Path(__file__).resolve().parents[1]
W=M/'work'
READER=M.parent/'mii-restoration-001/tools/zif.py'
assert hashlib.sha256(READER.read_bytes()).hexdigest()=='c2927450ed91212c21fcf665a732db82f3997f80668bfc071394fab5ff629269'
out=W/(sys.argv[1] if len(sys.argv)>1 else 'avatar-root-atlas-dds-probe-001')
assert out.parent==W and not out.exists()
out.mkdir()
sha=lambda b:hashlib.sha256(b).hexdigest()
rec=json.loads((M/'build/bake/BAKE-RECEIPT.json').read_bytes())
sources={}
for name in ['Avatar.zif','Avatar.zig','BAKE-RECEIPT.json']+[r['model']+'-atlas.png' for r in rec['models']]:
    src=M/'build/bake'/name
    data=src.read_bytes()
    sources[str(src)]=sha(data)
    (out/name).write_bytes(data)
shutil.copyfile(READER,out/'zif.py')
spec=importlib.util.spec_from_file_location('root_preserved_zif',out/'zif.py')
zmod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(zmod)
z=zmod.Zif(out/'Avatar.zif',out/'Avatar.zig')

def levels(raw):
    cur=np.asarray(raw,dtype=np.float64)
    result=[]
    while True:
        result.append(cur)
        if cur.shape[0]==cur.shape[1]==1:return result
        h,w=max(1,cur.shape[0]//2),max(1,cur.shape[1]//2)
        cur=cur.reshape(h,cur.shape[0]//h,w,cur.shape[1]//w,4).mean((1,3))

def encode(level):
    rounded=np.clip(np.floor(level+0.5),0,255).astype(np.uint8)
    h,w=rounded.shape[:2]
    rounded=np.pad(rounded,((0,max(0,4-h)),(0,max(0,4-w)),(0,0)),mode='edge')
    stream=io.BytesIO()
    Image.fromarray(rounded,'RGBA').save(stream,'DDS',pixel_format='DXT1')
    size=max(1,(h+3)//4)*max(1,(w+3)//4)*8
    return stream.getvalue()[128:128+size]

def decode(block,h,w,header):
    hd=bytearray(header)
    struct.pack_into('<4I',hd,12,h,w,len(block),0)
    struct.pack_into('<I',hd,28,1)
    struct.pack_into('<I',hd,108,0x1000)
    with Image.open(io.BytesIO(bytes(hd)+block)) as image:
        return np.array(image.convert('RGBA'),dtype=np.float64)

def sample(image,uv):
    h,w=image.shape[:2]
    x,y=np.clip(uv[:,0]*w-0.5,0,w-1),np.clip(uv[:,1]*h-0.5,0,h-1)
    x0,y0=np.floor(x).astype(int),np.floor(y).astype(int)
    x1,y1=np.minimum(x0+1,w-1),np.minimum(y0+1,h-1)
    fx,fy=(x-x0)[:,None],(y-y0)[:,None]
    return image[y0,x0]*(1-fx)*(1-fy)+image[y0,x1]*fx*(1-fy)+image[y1,x0]*(1-fx)*fy+image[y1,x1]*fx*fy

def mutation_differences(raw,cell,points,baseline,header):
    x,y,size=cell
    changed=np.asarray(raw).copy()
    keep=changed[y:y+size,x:x+size].copy()
    changed[:]=(255,0,0,255)
    changed[y:y+size,x:x+size]=keep
    result=[]
    for original,lv in zip(baseline,levels(changed)):
        reconstructed=decode(encode(lv),lv.shape[0],lv.shape[1],header)
        result.append(float(np.max(np.abs(sample(original,points)-sample(reconstructed,points)))))
    return result

results=[]
for mr in rec['models']:
    f,m=next((f,m) for f in z.forests for m in f['models'] if m['root']==mr['model'])
    rows=[r for r in z.meshes(f,m) if r['node']==0]
    assert len(rows)==1
    row=rows[0]
    vd=row['vertex_descriptor']
    assert vd[3]==32
    count,vo=vd[4],vd[5]
    uv=np.array([struct.unpack_from('<2e',z.g,vo+i*32+20) for i in range(count)],dtype=np.float64)
    uv[:,1]=1.0-uv[:,1]   # C253: stored V is native (1 - image V); the engine's shader samples at 1 - stored V
    tex=next(t for t in f['textures'] if t['pos']-f['base']==row['samplers'][0][1])
    tp=tex['header'][2]
    header=z.g[tp:tp+128]
    assert header[:4]==b'DDS ' and header[84:88]==b'DXT1'
    h,w=struct.unpack_from('<2I',header,12)
    mip_count=struct.unpack_from('<I',header,28)[0]
    raw=np.array(Image.open(out/(m['root']+'-atlas.png')).convert('RGBA'))
    assert raw.shape==(h,w,4) and h==w
    encoded=[encode(lv) for lv in levels(raw)]
    assert len(encoded)==mip_count
    payload=b''.join(encoded)
    assert z.g[tp+128:tp+128+len(payload)]==payload,'Actual DDS does not reconstruct from its recorded PNG'
    baseline=[decode(b,lv.shape[0],lv.shape[1],header) for b,lv in zip(encoded,levels(raw))]
    # No-gutter control built from the recorded cell contents; packed half-UVs are remapped correspondingly.
    control_raw=np.zeros_like(raw)
    control_raw[...,3]=255
    for part in mr['parts']:
        x,y,size=part['cell']['cell'];g=part['cell']['gutter'];c=part['cell']['content']
        content=raw[y+g:y+g+c,x+g:x+g+c]
        control_raw[y:y+size,x:x+size]=np.array(Image.fromarray(content,'RGBA').resize((size,size),Image.Resampling.BOX))
    control_baseline=[decode(encode(lv),lv.shape[0],lv.shape[1],header) for lv in levels(control_raw)]
    parts=[];offset=0
    for part in mr['parts']:
        n=part['vertices'];points=uv[offset:offset+n];offset+=n
        x,y,size=part['cell']['cell'];g=part['cell']['gutter'];c=part['cell']['content']
        assert n and np.isfinite(points).all()
        assert (points[:,0]>=(x-1)/w).all() and (points[:,0]<=(x+size+1)/w).all()
        assert (points[:,1]>=(y-1)/h).all() and (points[:,1]<=(y+size+1)/h).all(),'C253: sampled V leaves the chart cell'
        d=mutation_differences(raw,[x,y,size],points,baseline,header)
        local=np.clip((points*np.array([w,h])-np.array([x+g,y+g]))/c,0,1)
        control_points=(np.array([x,y])+local*size)/np.array([w,h])
        control_points[:,1]=1.0-control_points[:,1]                       # stored as native V in half float ...
        control_points=control_points.astype(np.float16).astype(np.float64)
        control_points[:,1]=1.0-control_points[:,1]                       # ... and sampled through the shader's inversion
        cd=mutation_differences(control_raw,[x,y,size],control_points,control_baseline,header)
        required=3 if size==64 else 4
        parts.append(dict(chart=part['chart'],vertices=n,cell=[x,y,size],required_isolated_through=required,
                          differences_per_mip=d,no_gutter_control_differences=cd,passed=all(v==0 for v in d[:required+1]),
                          isolated_through=next((k for k,v in enumerate(d) if v>0),len(d))-1))
    assert offset==count
    result=dict(model=m['root'],vertices=count,dds_sha256=sha(header+payload),dds_reconstructs_exactly=True,parts=parts,
                all_charts_meet_stated_mip_limits=all(p['passed'] for p in parts),
                no_gutter_control_leaks_at_mip0_or1=any(any(v>0 for v in p['no_gutter_control_differences'][:2]) for p in parts))
    results.append(result)
    print(json.dumps({k:v for k,v in result.items() if k!='parts'}),flush=True)
unchanged=all(sha(Path(p).read_bytes())==h for p,h in sources.items())
report=dict(status='ACTUAL_DDS_AND_HALF_UV_BOUNDARY_PROBE',python=sys.version,pillow=PIL.__version__,source_hashes=sources,
            parser_sha256=sha(READER.read_bytes()),inputs_unchanged_after_probe=unchanged,models=results,
            passed=unchanged and all(r['all_charts_meet_stated_mip_limits'] and r['no_gutter_control_leaks_at_mip0_or1'] for r in results),
            limits='Intermediate built files, no game/GPU. Exact DXT1 reconstruction and all recorded chart packed half-UVs. Bilinear samples use chart vertices, sufficient for axis-aligned cell separation; anisotropic sampling and live sampler behavior untested. Deeper mips retain recorded contamination and are not certified. No appearance/clothing acceptance implied.')
(out/'RECEIPT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(M/'tests/results/test_dds_isolation.json').write_text(json.dumps(dict(report,root_probe_source_sha256='8b74626a2c6cc80bd3f12067bfc12f995ccc54f1340d1481bb2ae7e75bd4ac8f',output=str(out)),indent=1)+'\n',encoding='utf-8')
print('PASS' if report['passed'] else 'FAIL',flush=True)
sys.exit(0 if report['passed'] else 1)
