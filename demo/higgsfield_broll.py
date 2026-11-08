#!/usr/bin/env python3
"""Generate the cinematic b-roll for the Ramp Model Labs demo with Higgsfield (Kling 2.6 Pro, text-to-video).

Needs two env vars from https://console.higgsfield.ai (API keys, funded balance):
    export HF_API_KEY_ID=...  HF_API_KEY_SECRET=...
Run:  python3 demo/higgsfield_broll.py          # writes demo/higgsfield/01-hook.mp4, 02-distill.mp4, 03-outro.mp4
Then: bash demo/assemble.sh                     # re-cuts the final video with the clips in place

Endpoint + schema per docs.higgsfield.ai/docs/models/kling-2-6/pro-text-to-video.md:
  POST https://api.higgsfield.ai/kling-video/v2.6/pro/text-to-video
  { prompt, duration: 5|10, aspect_ratio: "16:9"|"9:16"|"1:1", sound: "on"|"off", cfg_scale: 0..1 }
  -> { status, request_id, status_url, cancel_url }; poll status_url until status == "completed"; video.url
"""
import json, os, sys, time, uuid, urllib.request, pathlib

KEY_ID, KEY_SECRET = os.environ.get("HF_API_KEY_ID"), os.environ.get("HF_API_KEY_SECRET")
if not (KEY_ID and KEY_SECRET):
    sys.exit("set HF_API_KEY_ID and HF_API_KEY_SECRET (console.higgsfield.ai) first")

ENDPOINT = "https://api.higgsfield.ai/kling-video/v2.6/pro/text-to-video"
OUT = pathlib.Path(__file__).resolve().parent / "higgsfield"
OUT.mkdir(exist_ok=True)

# Palette: near-black ink (#0c0a08) and solar yellow (#e4f222). Abstract, no text, no logos, no faces.
SHOTS = [
    ("01-hook", 5,
     "Macro cinematic shot in a pure black void: thousands of glowing electric-yellow particles drift and converge "
     "toward a small bright hexagonal core, thin white wireframe geodesic lines faintly visible around it, volumetric "
     "bloom, shallow depth of field, slow dolly in, no text, no logo, premium tech launch film look"),
    ("02-distill", 5,
     "Abstract visualization of knowledge distillation on a black background: a huge luminous yellow sphere of light on "
     "the left exhales a thin stream of particles that crystallize into a tiny, sharp, perfectly formed hexagon on the "
     "right; smooth camera orbit, particles catch the light, elegant and minimal, no text"),
    ("03-outro", 5,
     "A single small glowing yellow hexagon floating in black space, hundreds of faint data points streaming past it at "
     "high speed like a highway at night, the hexagon stays perfectly still and sharp, slow push in, cinematic bloom, "
     "no text, no logo"),
]

def post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": f"Key {KEY_ID}:{KEY_SECRET}", "Content-Type": "application/json",
        "Idempotency-Key": str(uuid.uuid4())})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def get(url):
    req = urllib.request.Request(url, headers={"Authorization": f"Key {KEY_ID}:{KEY_SECRET}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

jobs = []
for name, seconds, prompt in SHOTS:
    target = OUT / f"{name}.mp4"
    if target.exists():
        print(f"skip {target.name} (exists)"); continue
    r = post(ENDPOINT, {"prompt": prompt, "duration": seconds, "aspect_ratio": "16:9", "sound": "off", "cfg_scale": 0.5})
    print(f"queued {name}: {r.get('request_id')}")
    jobs.append((name, target, r["status_url"]))

while jobs:
    time.sleep(8)
    for job in list(jobs):
        name, target, status_url = job
        st = get(status_url)
        s = st.get("status")
        if s == "completed":
            urllib.request.urlretrieve(st["video"]["url"], target)
            print(f"done   {name} -> {target}")
            jobs.remove(job)
        elif s in ("failed", "cancelled", "error"):
            print(f"FAILED {name}: {json.dumps(st)[:300]}"); jobs.remove(job)
        else:
            print(f"  {name}: {s}")
print("all clips in", OUT)
