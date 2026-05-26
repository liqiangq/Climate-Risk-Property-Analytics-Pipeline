# Climate-Risk-Property-Analytics-Pipeline

Data analytics, econometric modelling, machine learning, FastAPI, and a lightweight frontend for Napier property flood-risk analysis.

## Project Structure

- `backend/src/data_cleaning.py`: CSV loading, currency/date parsing, type cleanup, summary stats.
- `backend/src/feature_engineering.py`: log prices, time features, price index, flood-zone counts.
- `backend/src/modelling.py`: OLS, DiD-style OLS, and Ridge regression.
- `backend/src/visualization.py`: Matplotlib charts for the dashboard/API.
- `backend/main.py`: FastAPI app and API endpoints.
- `frontend/`: static dashboard served by FastAPI.
- `Empty_Land_Cleaned.csv`: cleaned data used by default.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Frontend Development

The frontend is a static dashboard served by FastAPI:

- Edit `frontend/index.html` for page structure.
- Edit `frontend/static/styles.css` for layout and visual design.
- Edit `frontend/static/app.js` for API calls and interactions.
- Keep the backend running with `uvicorn backend.main:app --reload`, then refresh the browser.

## API Examples

- `GET /api/summary`
- `GET /api/price-index?frequency=monthly`
- `GET /api/flood-zones?year=2020`
- `GET /api/model/did`
- `GET /api/model/ridge`
