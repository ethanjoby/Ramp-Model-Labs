# Product demo

`ramp-model-labs-demo.mp4` is a scripted screen recording of the real product, shot with
[AIOProductOS Studio](https://aioproductos.com/studio) (an MCP server that drives Playwright with an
on-screen cursor, camera zooms, callouts and title cards) over the song in the repo root.
`ramp-model-labs-demo-narrated.mp4` is the same cut with a local voice-over reading the captions;
`ramp-model-labs-demo.srt` has the captions as subtitles.

## Files

| File | What |
| --- | --- |
| `shotlist.mjs` | the shot list: six beats (hero, copilot, finished lab, playground, workflow, outro) |
| `assemble.sh` | joins the two takes, lays the music bed so the beat drops on the first reveal, mixes the narrated variant, merges subtitles, splices in Higgsfield b-roll when present |
| `higgsfield_broll.py` | generates the three cinematic b-roll clips with Higgsfield's API (Kling 2.6 Pro text-to-video); needs `HF_API_KEY_ID` and `HF_API_KEY_SECRET` from console.higgsfield.ai |
| `.work/` | gitignored working folder: the Studio driver, the demo stack's SQLite file, logged-in browser state, raw takes |

## Re-shooting

The recording runs against a throwaway stack so no real Google account is involved: the Firebase Auth
emulator on 9099, a backend on 8100 with its own database, and a production build of the frontend on
3100 (built with `API_URL=http://127.0.0.1:8100`, which Next bakes into the proxy at build time).

```bash
cd frontend && firebase emulators:start --only auth --project ramp-model-labs --config e2e/firebase.json &
cd backend && FIREBASE_AUTH_EMULATOR_HOST=127.0.0.1:9099 FRONTEND_ORIGIN=http://127.0.0.1:3100 \
  DATABASE_URL=sqlite:///$PWD/../demo/.work/labs.db SECRET_KEY=demo uv run uvicorn app.main:app --port 8100 &
cd frontend && NEXT_PUBLIC_FIREBASE_AUTH_EMULATOR=http://127.0.0.1:9099 NEXT_DIST_DIR=.next-e2e API_URL=http://127.0.0.1:8100 \
  npx next build && NEXT_PUBLIC_FIREBASE_AUTH_EMULATOR=http://127.0.0.1:9099 NEXT_DIST_DIR=.next-e2e npx next start -p 3100 &

cd demo/.work/studio-run
node signin.mjs                      # emulator sign-in → auth.json (Studio records with this logged-in state)
node seed.mjs gl fraud && node watch.mjs   # finished labs for the "minutes later" shot
node prep-workflow.mjs               # a finished workflow run for the Workflows shot

# take B first (idle machine), then take A (starts a real lab on camera), then assemble
STUDIO_OUT_DIR=../studio-output LAB_ID=lab_… WF_ID=wf_… N_CANDIDATES=… BEATS=lab,model,workflow,outro NAME=takeB VOICE=1 node driver.mjs ../../shotlist.mjs
STUDIO_OUT_DIR=../studio-output BEATS=hero,copilot NAME=takeA VOICE=1 node driver.mjs ../../shotlist.mjs
bash ../../assemble.sh
```

Drop Higgsfield clips into `demo/higgsfield/` (`01-hook.mp4`, `03-outro.mp4`) and re-run `assemble.sh` to get the
version with the cinematic open and close.
