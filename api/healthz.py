from fastapi import FastAPI

app = FastAPI()

# In Vercel, the function file path is the base. Use root route.
@app.get("/")
def health_check():
    return {"ok": True}