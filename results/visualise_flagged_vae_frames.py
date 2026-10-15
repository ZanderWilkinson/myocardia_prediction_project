"""Build a scrollable 3D gallery of frames flagged by the VAE preprocessing audit."""

import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training_data"
AUDIT_PATH = ROOT / "results" / "vae_frame_quality_audit.json"
HTML_PATH = ROOT / "results" / "vae_flagged_frames_gallery.html"
PER_STRUCTURE = 300

audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
metadata = pd.read_csv(DATA / "mixed_demographics.csv")
cards = []
total_flagged = 0
for patient in audit["flagged"]:
    index = int(patient["patient_index"])
    diagnosis = str(metadata.MI.iloc[index])
    excluded = bool(patient["excluded"])
    disposition = "Patient cycle excluded" if excluded else "Frame interpolated/repaired"
    cloud = np.load(DATA / "mixed_samples" / f"{index}{diagnosis}.npy", allow_pickle=False)
    for frame_no in patient["flagged_frames"]:
        frame = cloud[int(frame_no)]
        point_sets = []
        for structure in (0, 1, 2):
            xyz = frame[frame[:, 3] == structure, :3]
            if len(xyz) == 0:
                continue
            selected = np.linspace(0, len(xyz) - 1,
                                   min(PER_STRUCTURE, len(xyz)), dtype=int)
            point_sets.append((xyz[selected].astype("<f4", copy=False), structure))
        xyz = np.concatenate([part[0] for part in point_sets])
        regions = np.concatenate([
            np.full(len(part[0]), part[1], dtype=np.uint8) for part in point_sets
        ])
        packed = np.empty((len(xyz), 13), dtype=np.uint8)
        packed[:, :12] = xyz.view(np.uint8).reshape(-1, 12)
        packed[:, 12] = regions
        encoded = base64.b64encode(packed.tobytes()).decode("ascii")
        cards.append(
            f'<article class="card"><header><strong>Patient {index} · {diagnosis} · Frame {frame_no}</strong>'
            f'<span class="disposition {"excluded" if excluded else "repaired"}">{disposition}</span></header>'
            f'<canvas width="440" height="300" data-cloud="{encoded}" '
            f'aria-label="Patient {index}, frame {frame_no}, 3D spatial point cloud"></canvas></article>'
        )
        total_flagged += 1

template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Preprocessing flagged frames</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#18212b;font:15px system-ui,sans-serif}
.bar{position:sticky;top:0;z-index:2;background:#fff;padding:14px 22px;box-shadow:0 2px 8px #0002}
h1{font-size:20px;margin:0 0 5px}.note{color:#536170;font-size:13px}.controls{display:flex;gap:22px;align-items:center;margin-top:10px;flex-wrap:wrap}
label{font-size:13px}input{vertical-align:middle}.legend{display:inline-flex;gap:12px;margin-left:auto;font-size:12px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}.grid{padding:16px;display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:14px}
.card{background:#fff;border:1px solid #dce2e8;border-radius:8px;overflow:hidden;min-width:0}.card header{padding:8px 12px;font-size:13px;border-bottom:1px solid #edf0f3;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}
.disposition{font-size:11px;padding:2px 7px;border-radius:9px}.repaired{background:#e5f3eb;color:#236643}.excluded{background:#fff0dc;color:#85500d}
canvas{display:block;width:100%;height:auto;background:#fff}.footer{padding:18px;color:#64717e;text-align:center}
@media(max-width:760px){.grid{grid-template-columns:1fr;padding:9px;gap:9px}.legend{margin-left:0}}
</style></head><body><div class="bar"><h1>Frames flagged by VAE preprocessing</h1>
<div class="note">Development-set audit · original point clouds before preprocessing · 300 sampled points per anatomical structure. Drag a view to rotate; scroll to browse.</div>
<div class="controls"><label>Azimuth <input id="az" type="range" min="-180" max="180" value="35"></label>
<label>Elevation <input id="el" type="range" min="-70" max="70" value="22"></label>
<span id="count"></span><span class="legend"><span><i class="dot" style="background:#2878b5"></i>Region 0</span><span><i class="dot" style="background:#e07a2d"></i>Region 1</span><span><i class="dot" style="background:#3a9d67"></i>Region 2</span></span></div></div>
<main class="grid">__CARDS__</main><div class="footer">End of gallery</div>
<script>
const canvases=[...document.querySelectorAll('canvas')];document.querySelector('#count').textContent=`${canvases.length} flagged frame views`;
const colors=['#2878b5','#e07a2d','#3a9d67'];
function draw(c){if(!c.dataset.loaded)return;const ctx=c.getContext('2d'),w=c.width,h=c.height;
 const raw=atob(c.dataset.cloud),bytes=Uint8Array.from(raw,x=>x.charCodeAt(0)),n=bytes.length/13,view=new DataView(bytes.buffer),xyz=new Float32Array(n*3),ids=new Uint8Array(n);
 for(let i=0;i<n;i++){let b=i*13,o=i*3;xyz[o]=view.getFloat32(b,true);xyz[o+1]=view.getFloat32(b+4,true);xyz[o+2]=view.getFloat32(b+8,true);ids[i]=bytes[b+12]}
 const az=Number(document.querySelector('#az').value)*Math.PI/180,el=Number(document.querySelector('#el').value)*Math.PI/180,xs=[],ys=[],zs=[];
 for(let i=0;i<n;i++){let x=xyz[i*3],y=xyz[i*3+1],z=xyz[i*3+2],x1=x*Math.cos(az)-y*Math.sin(az),y1=x*Math.sin(az)+y*Math.cos(az);xs.push(x1);ys.push(y1*Math.cos(el)-z*Math.sin(el));zs.push(y1*Math.sin(el)+z*Math.cos(el))}
 let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys),span=Math.max(xmax-xmin,ymax-ymin)||1,scale=Math.min(w-44,h-44)/span;
 ctx.clearRect(0,0,w,h);ctx.fillStyle='#fff';ctx.fillRect(0,0,w,h);ctx.strokeStyle='#d8dee5';ctx.strokeRect(14,14,w-28,h-28);
 const order=Array.from({length:n},(_,i)=>i).sort((a,b)=>zs[a]-zs[b]);for(const i of order){ctx.globalAlpha=.72;ctx.fillStyle=colors[ids[i]];ctx.beginPath();ctx.arc(w/2+(xs[i]-(xmin+xmax)/2)*scale,h/2-(ys[i]-(ymin+ymax)/2)*scale,1.55,0,Math.PI*2);ctx.fill()}
 ctx.globalAlpha=1;ctx.fillStyle='#52606d';ctx.font='11px system-ui';ctx.fillText('3D projection · drag to rotate',18,h-8)}
function load(c){if(c.dataset.loaded)return;c.dataset.loaded='1';draw(c)}
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)load(e.target)}),{rootMargin:'300px'});canvases.forEach(c=>io.observe(c));
for(const id of ['az','el'])document.querySelector('#'+id).addEventListener('input',()=>canvases.forEach(c=>{if(c.dataset.loaded)draw(c)}));
let drag=null;canvases.forEach(c=>{c.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};c.setPointerCapture(e.pointerId)});c.addEventListener('pointermove',e=>{if(!drag)return;let dx=e.clientX-drag.x,dy=e.clientY-drag.y;drag={x:e.clientX,y:e.clientY};let a=document.querySelector('#az'),v=document.querySelector('#el');a.value=Math.max(-180,Math.min(180,+a.value+dx*.7));v.value=Math.max(-70,Math.min(70,+v.value-dy*.5));canvases.forEach(q=>{if(q.dataset.loaded)draw(q)})});c.addEventListener('pointerup',()=>drag=null);c.addEventListener('pointercancel',()=>drag=null)});
</script></body></html>'''
HTML_PATH.write_text(template.replace("__CARDS__", "\n".join(cards)), encoding="utf-8")
print(f"Saved {HTML_PATH}")
print(f"Rendered {total_flagged} flagged frames from {len(audit['flagged'])} patients.")
print(f"Audit: {audit['patients_checked']} development patients; {audit['frames_repaired']} frames repaired; "
      f"{audit['patients_excluded']} patient cycles excluded.")
