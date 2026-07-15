import os
from typing import TypedDict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Ensure API Key is configured
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "gq_your_placeholder_api_token")

# --- Structured Pydantic Schemas for Multi-Agent Consensus ---
class ExtractedDataModel(BaseModel):
    hcpName: Optional[str] = Field(None, description="Name of the Healthcare Professional/doctor")
    interactionType: Optional[str] = Field(None, description="Must be exactly: 'Meeting', 'Call', or 'Email'")
    topicsDiscussed: Optional[str] = Field(None, description="Key discussion points, product details, or feedback")
    outcomes: Optional[str] = Field(None, description="Agreed terms, commitments, or general conclusions reached")
    followUpActions: Optional[str] = Field(None, description="Next actionable steps or specific follow-up actions")

class SentimentModel(BaseModel):
    sentiment: str = Field(..., description="Must be exactly: 'Positive', 'Neutral', or 'Negative'")

class SuggestionsModel(BaseModel):
    suggestions: List[str] = Field(default=[], description="List of 2-3 specific pharmaceutical sales next steps")

# Define Graph State dict
class AgentState(TypedDict):
    message: str
    current_form_state: dict
    validation_error: Optional[str]
    extracted_data: dict
    sentiment_result: str
    suggestions_result: List[str]
    reply: str

# 1. Initialize LLMs (Gemma-2-9b-it for chat, Llama-3.3-70b-versatile for structured extraction)
gemma_chat = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY, temperature=0.5)
llama_extractor = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY, temperature=0.1)

# --- Agent 1: Validator Agent ---
def validator_agent(state: AgentState) -> dict:
    text = state["message"]
    # Check if user input is conversational noise or actually sales related
    if len(text.strip()) < 5:
         return {"validation_error": "Your entry is too short. Please describe the interaction details."}
    return {"validation_error": None}

# --- Agent 2: Core Entity Extractor Agent ---
def core_extractor_agent(state: AgentState) -> dict:
    if state.get("validation_error"):
        return {"extracted_data": {}}
    
    parser = llama_extractor.with_structured_output(ExtractedDataModel)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI Entity Extractor for a pharma CRM. Analyze the conversation and extract variables. Current values are: {current_state}. Do not overwrite valid values with blank spaces."),
        ("user", "{text}")
    ])
    chain = prompt | parser
    res = chain.invoke({"current_state": str(state["current_form_state"]), "text": state["message"]})
    
    # Filter non-null items
    extracted = {k: v for k, v in res.dict().items() if v is not None}
    return {"extracted_data": extracted}

# --- Agent 3: Sentiment Analyzer Agent ---
def sentiment_agent(state: AgentState) -> dict:
    if state.get("validation_error"):
         return {"sentiment_result": "Neutral"}
         
    parser = llama_extractor.with_structured_output(SentimentModel)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Analyze the clinical interaction context and classify the sentiment strictly into one of these buckets: 'Positive', 'Neutral', 'Negative'."),
        ("user", "{text}")
    ])
    chain = prompt | parser
    res = chain.invoke({"text": state["message"]})
    return {"sentiment_result": res.sentiment}

# --- Agent 4: Action Suggestions Agent ---
def next_action_agent(state: AgentState) -> dict:
    if state.get("validation_error"):
         return {"suggestions_result": []}
         
    parser = llama_extractor.with_structured_output(SuggestionsModel)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate 2 or 3 short, concrete follow-up tasks appropriate for pharmaceutical representatives based on this text. Start each with an action verb."),
        ("user", "{text}")
    ])
    chain = prompt | parser
    res = chain.invoke({"text": state["message"]})
    return {"suggestions_result": res.suggestions}

# --- Agent 5: Assistant Response Agent ---
def response_agent(state: AgentState) -> dict:
    if state.get("validation_error"):
        return {"reply": state["validation_error"]}
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional, encouraging AI CRM companion. Summarize to the sales rep what you parsed in under 2 sentences. Be cheerful and precise."),
        ("user", "{text}")
    ])
    chain = prompt | gemma_chat
    res = chain.invoke({"text": state["message"]})
    return {"reply": res.content}

# --- Router Logic Node ---
def routing_node(state: AgentState):
    if state.get("validation_error"):
        return "response_agent"
    return "core_extractor_agent"

# --- Assemble the StateGraph ---
workflow = StateGraph(AgentState)

# Add our 5 Nodes
workflow.add_node("validator_agent", validator_agent)
workflow.add_node("core_extractor_agent", core_extractor_agent)
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("next_action_agent", next_action_agent)
workflow.add_node("response_agent", response_agent)

# Configure Flow
workflow.set_entry_point("validator_agent")
workflow.add_conditional_edges("validator_agent", routing_node)
workflow.add_edge("core_extractor_agent", "sentiment_agent")
workflow.add_edge("sentiment_agent", "next_action_agent")
workflow.add_edge("next_action_agent", "response_agent")
workflow.add_edge("response_agent", END)

agent_executor = workflow.compile()
