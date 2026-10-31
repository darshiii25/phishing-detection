# Integration Layer: PhishGuard Frontend + Phishing-Detection Backend

This folder is a **separate integration layer**. It does **not** modify the
frontend or the backend. Both original codebases are treated as read-only.

The single problem this layer solves: the **frontend speaks HTTP** (it calls
`POST {baseUrl}/predict`) but the **backend is a CLI tool** with no web server.
The file `server.py` is a thin adapter that imports the backend's existing
`predict_url()` function and exposes it over an HTTP endpoint that exactly
matches the frontend's existing contract.

## What this layer does (and does not) do

- **Does**: import the backend's real prediction function (read-only use of
  backend code) and serve it over `POST /predict` with the response shape the
  frontend already expects.
- **Does**: lowercase the backend's label (`"Legitimate"`/`"Phishing"` →
  `"legitimate"`/`"phishing"`), because the frontend requires lowercase values.
- **Does**: send CORS headers, because the browser calls the adapter (port 8000)
  from the Vite dev-server origin (e.g. port 5173).
- **Does not**: fabricate results, mock data, or invent `reasons`/probabilities.
- **Does not**: edit, create, delete, or move any file inside the frontend or
  backend folders.
- **Does not**: add any third-party dependency. It uses only the Python standard
  library (`http.server`, `json`, `os`, `sys`, `threading`, `pathlib`).

## Prerequisites

The adapter must run in the **same Python environment as the backend**, because
it imports `tensorflow`, `numpy`, and the backend's modules. That environment
must already have the backend dependencies installed and the trained artifacts
present:

- `models/cnn_feature_extractor.keras`
- `models/lstm_classifier.keras`
- `outputs/vocab.json`
- `outputs/wso_selected_features.json`

If those artifacts are missing, re-run the backend training first:
`python -m src.train_pipeline` (from the backend root). See the backend README.

## How to run everything

Open three terminals.

### 1. (Nothing to start for the backend as a server)

The backend has no long-running server. Its models are loaded on demand by the
adapter below. Just make sure the backend Python environment (with tensorflow)
is the one you use to launch the adapter.

### 2. Start the integration adapter

From this `integration` folder, using the **backend's** Python/venv:

```powershell
# If you use a venv for the backend, activate it first, then:
python server.py
```

It listens on `http://127.0.0.1:8000` by default (matches the frontend default).
Override with environment variables if needed:

```powershell
$env:BACKEND_DIR = "C:\path\to\phishing-detection"
$env:INTEGRATION_PORT = "8000"
python server.py
```

Quick health check (does not load the model):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

### 3. Start the frontend

From the frontend folder:

```powershell
npm install   # only the first time
npm run dev
```

No `.env` file is required: the frontend already defaults
`VITE_API_BASE_URL` to `http://localhost:8000`, which is exactly where the
adapter listens. If you changed `INTEGRATION_PORT`, point the frontend at it by
creating a `.env` in the frontend folder:

```
VITE_API_BASE_URL=http://localhost:9000
```

(That `.env` would be a new file you create yourself; this integration layer does
not create it for you.)

## How it works

```
Browser  --POST /predict {url}-->  Adapter (server.py, :8000)
                                         |
                                         | imports predict_url() (read-only)
                                         v
                                   Backend (tensorflow models)
                                         |
                                         v  (label, legit_prob, phish_prob)
Browser  <--{prediction, *_probability}-- Adapter lowercases label,
                                          omits reasons (backend has none)
```

The frontend's `src/api.ts` validates that `prediction` is exactly `"phishing"`
or `"legitimate"` and that any probabilities are numbers in [0, 1]. The adapter
satisfies both. Because no `reasons` field is returned, the frontend shows its
built-in fallback: "Detailed reasons are not available for this prediction yet."

## Environment variables

| Variable          | Default                                      | Purpose                                |
| ----------------- | -------------------------------------------- | -------------------------------------- |
| `BACKEND_DIR`     | the known backend path used at integration   | Absolute path to the backend root       |
| `INTEGRATION_HOST`| `127.0.0.1`                                  | Host to bind                           |
| `INTEGRATION_PORT`| `8000`                                       | Port; matches frontend default         |

## Known limitations

- **Per-request model load**: the backend's `predict_url()` reloads the Keras
  models on every call (that is the backend's existing behavior). Each request
  may therefore take a few seconds. This is faithful to the backend and is not
  optimized here to avoid duplicating backend logic.
- **No `reasons`**: the backend does not produce explanation text, so none is
  returned. The frontend handles this case explicitly.
- **Single-machine / dev use**: the adapter is intended for local development.
  It uses permissive CORS (`*`) because there is no backend CORS setting to
  change and this layer is the only place headers can be added without editing
  either original codebase.
- **Live predictions require the backend environment**: if tensorflow or the
  model artifacts are unavailable, the adapter still starts but returns HTTP
  500 on `/predict` with the underlying error. No result is fabricated.
