from typing import TypedDict, List, Dict, Any, Literal
import json
import litellm
import logging
import pandas as pd
import json
import os
import glob

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Read .env before the configuration constants below are evaluated.
load_dotenv()

logger = logging.getLogger(__name__)

cwd_path = os.getcwd()
CLEANED_RBA_PATH = cwd_path + "/data/rba_cash_rate_cleaned.csv"
CLEANED_ASX_DIR = cwd_path + "/data/cleaned_asx"
CLEANED_AFR_DIR = cwd_path + "/data/cleaned_afr"

# --- LiteLLM Gateway Configuration ---
# Variable names follow Participant_Package/Setup_Instructions.md and
# handout/02_execution_guide.md. Those two documents disagree on the base-URL name
# (LITELLM_BASE_URL vs LITELLM_URL), so both are accepted.
# `or` chains rather than getenv defaults: an empty LITELLM_URL="" is a present
# value that a getenv default would not override.
LITELLM_BASE_URL = (
    os.getenv("LITELLM_BASE_URL")
    or os.getenv("LITELLM_URL")
    or "http://localhost:4000/v1"
)
LITELLM_KEY = os.getenv("LITELLM_KEY") or "sk-local-cluster"

# --- Model Roles (binding, per Challenge_Brief.md "Required Model Roles") ---
# BRAIN_MODEL plans and emits tool calls; DOMAIN_FT_MODEL synthesizes the final
# answer. The "openai/" prefix tells LiteLLM to treat the alias as an
# OpenAI-compatible route against LITELLM_BASE_URL.
BRAIN_MODEL = "openai/" + (os.getenv("BRAIN_MODEL") or "agent-brain")
DOMAIN_FT_MODEL = "openai/" + (os.getenv("DOMAIN_FT_MODEL") or "domain-ft")

# `mock` is the bootstrap plumbing mode and must be switched to `llm` before
# evaluation, otherwise the fine-tuned model is not actually used and the
# submission loses model-quality and architecture credit.
DOMAIN_PREDICT_MODE = (os.getenv("DOMAIN_PREDICT_MODE") or "llm").strip().lower()

# Maximum tool iterations before the graph is forced to synthesize.
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS") or "5")


def call_llm(model: str, system_prompt: str, user_message: str) -> str:
    """Send a single-turn completion through the LiteLLM gateway and return the text."""
    response = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.0,
        api_base=LITELLM_BASE_URL,
        api_key=LITELLM_KEY
    )
    return response.choices[0].message.content

# --- 1. State Definition ---
class AgentState(TypedDict):
    question: str
    plan: str
    tool_calls: List[Dict[str, Any]]
    tool_outputs: List[Dict[str, Any]]
    context: str
    final_answer: str
    loop_count: int
    max_loops: int
    tool_trace: List[Dict[str, Any]]


# --- 2. Aligned Mock Tools (Using Cleaned Schemas) ---
def query_rba_cash_rate(effective_date: str = None) -> str:
    """
    Queries the local RBA cash rate decisions dataset.
    :param effective_date: Optional. Specific date formatted as YYYY-MM-DD. 
                           If omitted, or set to "all" / "history", returns the entire historical dataset.
    """
    if not os.path.exists(CLEANED_RBA_PATH):
        return f"Error: Cleaned RBA database file not found at {CLEANED_RBA_PATH}."

    try:
        # Load cleaned database
        df = pd.read_csv(CLEANED_RBA_PATH)
        
        # Ensure effective_date column matches string-wise
        df["effective_date"] = df["effective_date"].astype(str).str.strip()

        # Check if the model is requesting the entire history
        if not effective_date or effective_date.lower().strip() in ["all", "history", "none", "null"]:
            # Convert the entire dataframe to a compact list of dicts
            records = df.to_dict(orient="records")
            return json.dumps(records)
        
        # Match exact date
        target_date = effective_date.strip()
        result_df = df[df["effective_date"] == target_date]
        
        if not result_df.empty:
            record = result_df.iloc[0].to_dict()
            
            # Return serialized JSON string of the record
            return json.dumps({
                "effective_date": str(record["effective_date"]),
                "cash_rate_target": float(record["cash_rate_target"]),
                "change_pct": float(record["change_pct"]),
                "change_bps": int(record["change_bps"])
            })
        else:
            return f"No RBA decision found for date {effective_date}."

    except Exception as e:
        return f"Error accessing local RBA dataset: {str(e)}"

# def query_rba_cash_rate(effective_date: str) -> str:
#     """
#     Search the RBA cash-rate dataset.
#     :param effective_date: Date formatted as YYYY-MM-DD.
#     """
#     # Mock lookup using the standardized format we compiled
#     if effective_date == "2010-02-03":
#         record = {
#             "effective_date": "2010-02-03",
#             "cash_rate_target": 3.75,
#             "change_pct": 0.00,
#             "change_bps": 0
#         }
#         return json.dumps(record)
#     return f"No RBA decision found for date {effective_date}."


# def query_asx_prices(ticker: str, date: str) -> str:
#     """
#     Search the ASX historical price database.
#     :param ticker: Cleaned stock ticker (e.g., 'AGL', not 'AGL.AX').
#     :param date: Date formatted as YYYY-MM-DD.
#     """
#     # Mock lookup using the rounded, standardized format
#     if ticker.upper() == "AGL" and date == "2015-01-02":
#         record = {
#             "ticker": "AGL",
#             "date": "2015-01-02",
#             "open": 7.63,
#             "high": 7.63,
#             "low": 7.48,
#             "close": 7.55,
#             "volume": 359519
#         }
#         return json.dumps(record)
#     return f"No ASX prices found for {ticker} on {date}."

def query_asx_prices(ticker: str, date: str) -> str:
    """
    Queries the partitioned local cleaned ASX dataset.
    :param ticker: The stock ticker (e.g. 'IAG')
    :param date: Date formatted as YYYY-MM-DD (e.g. '2015-01-02')
    """
    # Normalize ticker input to uppercase for filename matching
    clean_ticker = ticker.strip().replace(".AX", "").replace(".ax", "").upper()
    target_date = date.strip()
    
    # Path to the specific ticker file
    ticker_file_path = os.path.join(CLEANED_ASX_DIR, f"{clean_ticker}.jsonl")
    
    if not os.path.exists(ticker_file_path):
        return f"No ASX record found. Ticker '{clean_ticker}' does not exist in local dataset."

    try:
        # Stream-read only the specific ticker file line-by-line
        with open(ticker_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                record = json.loads(line)
                if record.get("date") == target_date:
                    # Return the exact matched clean row
                    return json.dumps({
                        "ticker": record["ticker"],
                        "date": record["date"],
                        "open": record["open"],
                        "high": record["high"],
                        "low": record["low"],
                        "close": record["close"],
                        "volume": record["volume"]
                    })
                    
        return f"No stock prices found for ticker '{clean_ticker}' on date {target_date}."
        
    except Exception as e:
        return f"Error reading ASX record for {clean_ticker}: {str(e)}"

# def query_afr_news(query: str, date_filter: str = None) -> str:
#     """
#     Search the AFR news corpus for articles.
#     :param query: Keywords or company names to search.
#     :param date_filter: Optional publication date formatted as YYYY-MM-DD.
#     """
#     # Mock lookup returning cleaned text structures
#     if "BC Iron" in query:
#         record = {
#             "headline": "BC Iron tips short-term price relief",
#             "intro": "BC Iron managing director Morgan Ball says he expects the iron ore price to rebound...",
#             "text": "BC Iron managing director... Nullagine joint venture shipped 1.38 million wet metric tonnes...",
#             "newspaper": "Australian Financial Review",
#             "publication_date": "2015-01-31"
#         }
#         return json.dumps(record)
#     return f"No AFR news results for search: '{query}'."

def query_afr_news(query: str, date_filter: str = None, max_results: int = 3) -> str:
    """
    Scans the cleaned local AFR news corpus folder for articles containing the query terms.
    :param query: Search terms or company names (case-insensitive).
    :param date_filter: Optional date filter formatted as YYYY-MM-DD.
    :param max_results: Maximum number of matching articles to return to the agent.
    """
    if not os.path.exists(CLEANED_AFR_DIR):
        return f"Error: Cleaned AFR directory not found at {CLEANED_AFR_DIR}."

    search_terms = query.lower().strip().split()
    if not search_terms:
        return "Error: Empty search query provided."

    target_files = glob.glob(os.path.join(CLEANED_AFR_DIR, "*.jsonl"))
    matches = []

    try:
        for file_path in target_files:
            if len(matches) >= max_results:
                break

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    record = json.loads(line)
                    
                    # 1. Apply Date Filter (if specified)
                    if date_filter and record.get("publication_date") != date_filter.strip():
                        continue

                    # 2. Extract text fields for text searching
                    headline = record.get("headline", "").lower()
                    intro = record.get("intro", "").lower()
                    text = record.get("text", "").lower()

                    # 3. Perform term matching (logical AND for all terms in query)
                    is_match = all(term in headline or term in intro or term in text for term in search_terms)

                    if is_match:
                        matches.append({
                            "headline": record["headline"],
                            "publication_date": record["publication_date"],
                            "intro": record["intro"],
                            "text": record["text"][:800] + "..." if len(record["text"]) > 800 else record["text"] # Truncate body text to conserve tokens
                        })
                        if len(matches) >= max_results:
                            break

        if matches:
            return json.dumps(matches)
        else:
            date_info = f" on date {date_filter}" if date_filter else ""
            return f"No AFR news articles found matching search query '{query}'{date_info}."

    except Exception as e:
        return f"Error searching AFR news database: {str(e)}"

# --- 3. Dynamic Tool Router ---
def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Routes execution calls to the appropriate cleaned Python functions."""
    try:
        if tool_name == "query_rba_cash_rate":
            return query_rba_cash_rate(arguments.get("effective_date"))
        elif tool_name == "query_asx_prices":
            return query_asx_prices(arguments.get("ticker", ""), arguments.get("date", ""))
        elif tool_name == "query_afr_news":
            return query_afr_news(arguments.get("query", ""), arguments.get("date_filter"))
        else:
            return f"Error: Tool '{tool_name}' not recognized."
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


# --- 4. Graph Nodes ---

def reasoning_planner(state: AgentState) -> Dict[str, Any]:
    """
    Qwen-35B acts as the planner. It is now explicitly instructed
    to use the standardized ISO date formats and cleaned parameters.
    """
    question = state["question"]
    context = state["context"]
    loop_count = state.get("loop_count", 0)

    system_prompt = system_prompt = (
    "You are an expert financial planning assistant. You have access to three local database tools:\n\n"
    "1. query_rba_cash_rate(effective_date: str) -> Expects YYYY-MM-DD. "
    "   IMPORTANT: If a question requires counting, scanning, or identifying historical trends across "
    "   the entire dataset, pass effective_date=all to retrieve the full history.\n"
    "2. query_asx_prices(ticker: str, date: str) -> Expects clean ticker (e.g., 'AGL') and date as YYYY-MM-DD\n"
    "3. query_afr_news(query: str, date_filter: str) -> Searches news, optional date as YYYY-MM-DD\n\n"
    
    "CRITICAL INSTRUCTIONS:\n"
    "- You must think step-by-step before taking an action. Write your step-by-step plan inside <thinking>...</thinking> tags.\n"
    "- Following the thinking block, you must output exactly one action line: either a tool CALL or a final DECISION.\n"
    "- Do not write any conversational filler outside the <thinking> block.\n\n"
    
    "EXAMPLES:\n\n"
    
    "Example 1: Analyzing the entire history (Easy/Aggregate)\n"
    "User Question: From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The user is asking for aggregate statistics (counts of changes, increases, and decreases) over the entire historical range of the RBA dataset.\n"
    "2. This cannot be solved with a point-lookup of a single date.\n"
    "3. I must retrieve the complete RBA history to allow the synthesis model to perform the counts.\n"
    "4. I will call the RBA tool with effective_date=all.\n"
    "</thinking>\n"
    "CALL: query_rba_cash_rate|effective_date=all\n\n"
    
    "Example 2: Analyzing a specific temporal range (Medium/Cycle)\n"
    "User Question: Across the 2011-2013 easing period, how many cuts occurred and how far did the target fall?\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The user wants to analyze a specific temporal window (2011-2013 easing period).\n"
    "2. To count the cuts and identify the start/end rate targets over this period, I must inspect the historical sequence of records.\n"
    "3. Retrieving the complete RBA dataset is the most reliable way to let the synthesis engine filter, count, and sum target movements between 2011 and 2013.\n"
    "4. I will call the RBA tool with effective_date=all.\n"
    "</thinking>\n"
    "CALL: query_rba_cash_rate|effective_date=all\n\n"
    
    "Example 3: Querying the AFR news on a specific event\n"
    "User Question: What were the key points of the coroner's decision regarding the Downer EDI inquest mentioned in early 2015?\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The user is asking about a specific news event involving Downer EDI and a coroner's decision in early 2015.\n"
    "2. I need to search the AFR news corpus for relevant articles using keywords like 'Downer EDI' and 'coroner' or 'Alec Meikle'.\n"
    "3. I will call the query_afr_news tool with an appropriate query string.\n"
    "</thinking>\n"
    "CALL: query_afr_news|query=Downer EDI coroner\n\n"
    
    "Example 4: Ready to Synthesize\n"
    "User Question: From the first RBA record to the last, how many cash-rate decisions changed the rate, and how many were increases versus decreases?\n"
    "Accumulated Data Context:\n"
    "[{\"effective_date\": \"2010-02-03\", \"change_pct\": 0.00, \"cash_rate_target\": 3.75, \"change_bps\": 0}, ... (all 175 records present)]\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. I have successfully retrieved the entire RBA dataset, which contains all 175 historical record rows.\n"
    "2. The synthesis model has sufficient context to count the non-zero changes and segment them into increases (positive changes) and decreases (negative changes).\n"
    "3. No more tool queries are required. I will transition to the synthesis phase.\n"
    "</thinking>\n"
    "DECISION: READY"
)

    user_message = f"User Question: {question}\n\nAccumulated Data Context:\n{context}\n\nDetermine the next step."

    response_text = call_llm(BRAIN_MODEL, system_prompt, user_message)
    
    # Tool parser
    tool_calls = []
    if "CALL:" in response_text:
        try:
            line = [l for l in response_text.split("\n") if "CALL:" in l][0]
            parts = line.replace("CALL:", "").strip().split("|")
            tool_name = parts[0].strip()
            args = {}
            if len(parts) > 1:
                for pair in parts[1].split(","):
                    if "=" in pair:
                        k, v = pair.split("=")
                        args[k.strip()] = v.strip()
            tool_calls.append({"name": tool_name, "arguments": args})
        except Exception:
            pass  # Fallback handled by empty tool_calls array

    return {
        "plan": response_text,
        "tool_calls": tool_calls,
        "loop_count": loop_count + 1
    }


def execute_tools_node(state: AgentState) -> Dict[str, Any]:
    """Executes tools and appends metadata to cumulative tool_trace."""
    tool_calls = state["tool_calls"]
    current_context = state["context"]
    current_trace = state.get("tool_trace", []) or []

    outputs = []
    new_trace_entries = []

    for call in tool_calls:
        output = execute_tool(call["name"], call["arguments"])
        outputs.append(output)
        
        # Trace shape matches the README "Question Endpoint" example: tool/args/result.
        new_trace_entries.append({
            "tool": call["name"],
            "args": call["arguments"],
            "result": output
        })
        
    combined_new_context = "\n".join(outputs)
    updated_context = f"{current_context}\n{combined_new_context}".strip() if current_context else combined_new_context

    return {
        "tool_outputs": outputs,
        "context": updated_context,
        "tool_calls": [],  # Clear current queue
        "tool_trace": current_trace + new_trace_entries
    }


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """
    The fine-tuned Nemotron domain model synthesizes the final grounded answer
    from the verified tool results.
    """
    question = state["question"]
    context = state["context"]

    if DOMAIN_PREDICT_MODE == "mock":
        # Bootstrap plumbing mode only -- the fine-tuned model is bypassed.
        logger.warning(
            "DOMAIN_PREDICT_MODE=mock: returning a stub answer without calling %s. "
            "Set DOMAIN_PREDICT_MODE=llm before evaluation.",
            DOMAIN_FT_MODEL
        )
        return {"final_answer": f"[mock] Context gathered for: {question}\n{context}".strip()}

    system_prompt = (
        "You are an expert financial analysis synthesizer. Generate a direct, grounded answer "
        "to the question based strictly on the provided context (which is formatted as JSON blocks)."
    )

    user_message = f"Context Blocks:\n{context}\n\nQuestion: {question}"

    return {"final_answer": call_llm(DOMAIN_FT_MODEL, system_prompt, user_message)}


# --- 5. Conditional Routing ---
def router(state: AgentState) -> Literal["execute_tools", "synthesize_answer"]:
    if state["loop_count"] >= state["max_loops"]:
        return "synthesize_answer"
    
    plan = state.get("plan", "")
    if "DECISION: READY" in plan or not state.get("tool_calls"):
        return "synthesize_answer"
        
    return "execute_tools"


# --- 6. Build the Graph Workflow ---
workflow = StateGraph(AgentState)

workflow.add_node("reasoning_planner", reasoning_planner)
workflow.add_node("execute_tools", execute_tools_node)
workflow.add_node("synthesize_answer", synthesis_node)

workflow.add_edge(START, "reasoning_planner")
workflow.add_conditional_edges(
    "reasoning_planner",
    router,
    {
        "execute_tools": "execute_tools",
        "synthesize_answer": "synthesize_answer"
    }
)
workflow.add_edge("execute_tools", "reasoning_planner")
workflow.add_edge("synthesize_answer", END)

app = workflow.compile()


# --- 7. Execution Endpoint ---
def run_financial_agent(question: str, max_loops: int = MAX_AGENT_STEPS) -> Dict[str, Any]:
    """
    Executes the pipeline and returns the structure demanded by the scoring system.
    """
    initial_state = {
        "question": question,
        "plan": "",
        "tool_calls": [],
        "tool_outputs": [],
        "context": "",
        "final_answer": "",
        "loop_count": 0,
        "max_loops": max_loops,
        "tool_trace": []
    }
    
    final_state = app.invoke(initial_state)
    
    return {
        "answer": final_state.get("final_answer", ""),
        "steps": final_state.get("loop_count", 0),
        "tool_trace": final_state.get("tool_trace", [])
    }
