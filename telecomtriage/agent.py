import os
import requests
import chromadb
from chromadb.api.types import EmbeddingFunction
from google import genai
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
import sqlite3
from .guardrails import refund_amount_guard



# For Vector Database set up 
client_genai = genai.Client()

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    def __call__(self, input):
        result = client_genai.models.embed_content(
            model="gemini-embedding-001",
            contents=input,
        )
        return [e.values for e in result.embeddings]

gemini_ef = GeminiEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./chroma_db_recap")
kb_collection = chroma_client.get_or_create_collection(name="company_docs", embedding_function=gemini_ef)

# Seed it once — skip if already populated
if kb_collection.count() == 0:
    kb_collection.add(
        documents=[
            "Our return policy allows refunds within 30 days of purchase with a receipt.",
            "Standard shipping takes 5-7 business days. Express shipping takes 2 days.",
            "Customer support is available Monday-Friday, 9am-6pm EST, via chat or email.",
        ],
        ids=["doc1", "doc2", "doc3"],
    )

print(f"Knowledge base has {kb_collection.count()} documents.")


# Search Policy Docs Function
def search_policy_docs(query: str) -> str:
    """Search internal company documents for policy, shipping, or support information."""
    results = kb_collection.query(query_texts=[query], n_results=2)
    matches = results["documents"][0]
    if not matches:
        return "No relevant information found in the knowledge base."
    return "\n".join(matches)



# SQLite database
# One-time setup: create a fake but realistic billing table
conn = sqlite3.connect("billing.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS accounts (
        account_id TEXT PRIMARY KEY,
        plan_type TEXT,
        current_bill REAL,
        outage_minutes_last_cycle INTEGER
    )
""")
conn.execute("INSERT OR REPLACE INTO accounts VALUES ('ABC123', 'unlimited', 95.50, 180)")
conn.execute("INSERT OR REPLACE INTO accounts VALUES ('DEF123', 'limited', 65.50, 50)")
conn.commit()

# Account Usage Function
def get_account_usage(account_id: str) -> str:
    """Look up billing and usage details for a customer account."""
    account_id = account_id.strip().upper()
    conn = sqlite3.connect("billing.db")
    row = conn.execute(
        "SELECT plan_type, current_bill, outage_minutes_last_cycle FROM accounts WHERE account_id = ?",
        (account_id,)
    ).fetchone()
    if not row:
        return f"No account found with ID {account_id}."
    plan, bill, outage_min = row
    return f"Account {account_id}: {plan} plan, current bill ${bill}, {outage_min} minutes of outage last cycle."

# Refund Check Function
def check_refund_eligibility(outage_minutes: int, plan_type: str) -> str:
    """Determine if a customer qualifies for a refund based on outage duration and plan type."""
    threshold = 60 if plan_type == "unlimited" else 120
    if outage_minutes >= threshold:
        return f"Eligible for a refund — outage exceeded the {threshold}-minute threshold for {plan_type} plans."
    return f"Not eligible — outage was under the {threshold}-minute threshold for {plan_type} plans."


# Agent Code 

# Specialist 1: Billing Expert
billing_Agent = Agent(
    name="billing_specialist",
    model="gemini-3.6-flash",
    description="Handles queries about account and refund questions.",
    instruction="""
    You are a billing representative. Safely analyze customer accounts
    Never promise exact refund amounts without checking system boundaries or human-in-loop
    """,
    tools=[get_account_usage, check_refund_eligibility],
    after_model_callback=refund_amount_guard,
)

# Specialist 2: Policy Expert
policy_Agent = Agent(
    name="policy_specialist",
    model="gemini-3.6-flash",
    description="Handles queries about policy/shipping/support.",
    instruction="""
    You are a policy representative. You handle policy/shipping/support questions.
    """,
    tools=[search_policy_docs],
    
)

# Generalist : Router Agent
root_agent = Agent(
    name="router_agent",
    model="gemini-3.6-flash",
    description="An agent that routes queries to specialist agent.",
    instruction="""
    You are the generalist router agent. Your task is to analyze user queries.
    - If they describe a bill problem, delegate to billing_Agent.
    - If they describe a policy problem , delegate to policy_Agent.
    """,
    tools=[AgentTool(agent=billing_Agent),AgentTool(agent=policy_Agent)],
    after_model_callback=refund_amount_guard,
)

