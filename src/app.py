from fastapi import FastAPI

app = FastAPI(title="Global Agents API")

@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "healthy"}