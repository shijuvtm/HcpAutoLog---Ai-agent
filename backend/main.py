from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agent import agent_executor

app = FastAPI(title="Log HCP Interaction - API Gateway")

# Enable CORS for React frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatPayload(BaseModel):
    message: str
    current_form_state: dict
    history: List[dict]

class InteractionSubmitPayload(BaseModel):
    hcpName: str
    interactionType: str
    date: str
    time: str
    topicsDiscussed: Optional[str] = ""
    sentiment: str
    outcomes: Optional[str] = ""
    followUpActions: Optional[str] = ""
    aiSuggestions: List[str] = []

@app.post("/api/chat-agent")
async def chat_and_fill_form(payload: ChatPayload):
    try:
        inputs = {
            "message": payload.message,
            "current_form_state": payload.current_form_state,
            "chat_history": payload.history,
            "validation_error": None,
            "extracted_data": {},
            "sentiment_result": "",
            "suggestions_result": [],
            "reply": ""
        }
        
        output = agent_executor.invoke(inputs)
        
        # Merge sentiment safely into extracted data
        extracted = output.get("extracted_data", {})
        if output.get("sentiment_result"):
            extracted["sentiment"] = output["sentiment_result"]
            
        return {
            "reply": output["reply"],
            "extractedData": extracted,
            "suggestions": output.get("suggestions_result", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph Error: {str(e)}")

@app.post("/api/interactions")
async def save_interaction(payload: InteractionSubmitPayload):
    # Process structured submission into your DB engine here
    return {"status": "success", "message": "Interaction saved successfully to the CRM Database!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
