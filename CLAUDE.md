# SideEffectViz

Interactive pharmacovigilance app that visualizes medication side effects using FDA FAERS data.

## Deployment

- **Platform**: Railway (auto-deploys from `main` branch on GitHub)
- **Repo**: https://github.com/nickjlamb/sideeffectviz_streamlit
- **Build**: Nixpacks (configured in `railway.toml`)
- **Start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

## Project structure

- `app.py` — The entire application (self-contained, no local imports)
- `requirements.txt` — Python dependencies
- `railway.toml` — Railway deployment config
- `Procfile` — Process definition for Railway

### Local-only reference files (not deployed)

- `data_acquisition.py` — Standalone FAERS data download/processing script
- `ml_components.py` — Standalone ML clustering module
- `visualizations.py` — Standalone visualization module
- `refinement_and_scaling.md` — Development roadmap/notes

## Key constraints

- **OpenFDA API limit**: Max 100 results per request. Never set `limit` param above 100.
- App falls back to `generate_sample_data()` if the API call fails.

## Running locally

```
streamlit run app.py
```
