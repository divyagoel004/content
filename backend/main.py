from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from logic import generate_presentation_slides, export_to_pdf
from fastapi.responses import Response
import os

app = FastAPI()

# Mount the static directory to serve images
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

class GenerationRequest(BaseModel):
    topic: str
    depth_level: str = "intermediate"

@app.post("/api/generate")
async def generate(request: GenerationRequest):
    try:
        presentation_data = generate_presentation_slides(request.topic, request.depth_level)
        # Convert Pydantic models to dicts for JSON response
        return presentation_data
    except Exception as e:
        # Log the exception for debugging
        print(f"Error during presentation generation: {e}")
        # Return a meaningful error to the client
        raise HTTPException(status_code=500, detail=f"An error occurred during presentation generation: {e}")

class PdfRequest(BaseModel):
    presentation_data: dict

@app.post("/api/export/pdf")
async def export_pdf(request: PdfRequest):
    try:
        pdf_bytes = export_to_pdf(request.presentation_data)
        if pdf_bytes:
            return Response(content=pdf_bytes, media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=presentation.pdf"})
        else:
            raise HTTPException(status_code=500, detail="Failed to generate PDF.")
    except Exception as e:
        print(f"Error during PDF export: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during PDF export: {e}")

@app.get("/")
def read_root():
    return {"message": "Backend is running"}
