import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import uvicorn


from routes import videoprocess_api

# Initialize FastAPI app
app = FastAPI()

# Mount static files directory securely
app.mount("/videos/result", StaticFiles(directory="videos/result"), name="videos_result")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


app.include_router(videoprocess_api.router)

@app.get("/")
async def root():
    try:
        return {"message": "Welcome to the AI Backend API"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)