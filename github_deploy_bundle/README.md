# Youth Mental Health – Australia (Dash)

This repository hosts a Dash app. You can deploy it via **Render**, **Railway**, or **Heroku** (using GitHub integration).

## Local dev
```bash
pip install -r requirements.txt
python app.py
```

## Deploy (Render)
1. Push this repo to GitHub.
2. In Render: *New +* → **Web Service** → Connect your repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app.server`
5. Instance: Free tier is fine for demos.
6. After deploy, share the Render URL.

## Deploy (Heroku)
1. Ensure `requirements.txt`, `Procfile`, `runtime.txt` exist.
2. In Heroku: Create new app → Deploy via **GitHub** → Connect repo → Deploy branch.
3. Open app and share the URL.

## Codespaces (temporary sharing)
- Open the repo in **GitHub Codespaces**.
- Run `python app.py` and set forwarded Port **8050** to **Public** to share a preview link.
