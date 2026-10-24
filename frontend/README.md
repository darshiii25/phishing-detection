# PhishGuard frontend

Single-page React + TypeScript URL scanner UI. The Python repository is kept separate and unchanged.

## Run locally

From this directory:

```powershell
npm install
Copy-Item .env.example .env.local
npm run dev
```

Vite prints the local preview URL. `npm run build` creates a production build in `dist/`.

## Prediction service contract

The Python project's `api_server.py` is a small local HTTP adapter around the existing `src/predict_url.py` helper. It does not change the model code or train anything. Start it in a second terminal from the `phishing-detection/` directory with `python api_server.py`; it listens at `http://127.0.0.1:8000`. The UI sends requests only to the configured service and never substitutes a sample prediction when a request fails.

Set `VITE_API_BASE_URL` in `.env.local` (default: `http://localhost:8000`). Implement `POST /predict` in a separate backend adapter when ready:

```json
{ "url": "https://example.com" }
```

Return JSON with a supported lowercase prediction label; probabilities are optional, must be numbers from 0 through 1, and are shown only when provided. `reasons` is an optional array of strings containing evidence from the model or validated analysis.

```json
{
  "prediction": "phishing",
  "legitimate_probability": 0.18,
  "phishing_probability": 0.82,
  "reasons": ["Evidence observed by the analysis"]
}
```

The frontend has a separate **View sample result** control for UI preview. The result is prominently marked as illustrative demo data and is never presented as the output of Analyze URL. The app does not navigate to or fetch the inspected URL.
