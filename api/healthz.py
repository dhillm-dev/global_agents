from fastapi import FastAPI

app = FastAPI(title="Healthz")

@app.get("/")
def health():
    return {"ok": True}