from typing import TypedDict, List, Dict, Any, Literal
import json
import litellm
import logging
import pandas as pd
import os
import glob

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Read .env before the configuration constants below are evaluated.
load_dotenv()

# Imported after load_dotenv() so the LANGSMITH_* settings are visible to it.
from src.tracing import TRACING_ENABLED, traceable  # noqa: E402

logger = logging.getLogger(__name__)

cwd_path = os.getcwd()
CLEANED_RBA_PATH = cwd_path + "/data/rba_cash_rate_cleaned.csv"
CLEANED_ASX_DIR = cwd_path + "/data/cleaned_asx"
CLEANED_AFR_DIR = cwd_path + "/data/cleaned_afr"

# --- LiteLLM Gateway Configuration ---
LITELLM_BASE_URL = (
    os.getenv("LITELLM_BASE_URL")
    or os.getenv("LITELLM_URL")
    or "http://localhost:4000/v1"
)
LITELLM_KEY = os.getenv("LITELLM_KEY") or "sk-local-cluster"

# --- Model Roles ---
BRAIN_MODEL = "openai/" + (os.getenv("BRAIN_MODEL") or "agent-brain")
DOMAIN_FT_MODEL = "openai/" + (os.getenv("DOMAIN_FT_MODEL") or "domain-ft")

DOMAIN_PREDICT_MODE = (os.getenv("DOMAIN_PREDICT_MODE") or "llm").strip().lower()

# Maximum tool iterations before the graph is forced to synthesize.
MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS") or "5")


@traceable(run_type="llm", name="litellm_completion")
def call_llm(model: str, system_prompt: str, user_message: str) -> str:
    """Send a single-turn completion through the LiteLLM gateway and return the text.

    Traced explicitly: these are raw LiteLLM calls, not LangChain models, so
    LangGraph's automatic instrumentation does not capture them.
    """
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


# --- 1. State Definition (Added error_message) ---
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
    error_message: str  # Tracks execution exceptions for fallback synthesis


# --- 2. Tool Retrieval Definitions ---
def query_rba_cash_rate(effective_date: str = None) -> str:
    """Queries the local RBA cash rate decisions dataset.
    :param effective_date: Optional. Specific date formatted as YYYY-MM-DD. 
                           If omitted, or set to "all" / "history", returns the entire historical dataset.
    """
    if not os.path.exists(CLEANED_RBA_PATH):
        return f"Error: Cleaned RBA database file not found at {CLEANED_RBA_PATH}."

    try:
        df = pd.read_csv(CLEANED_RBA_PATH)
        df["effective_date"] = df["effective_date"].astype(str).str.strip()

        if not effective_date or effective_date.lower().strip() in ["all", "history", "none", "null"]:
            records = df.to_dict(orient="records")
            return json.dumps(records)
        
        target_date = effective_date.strip()
        result_df = df[df["effective_date"] == target_date]
        
        if not result_df.empty:
            record = result_df.iloc[0].to_dict()
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

def query_asx_prices(ticker: str = None, date: str = None) -> str:
    """
    Queries the partitioned local cleaned ASX dataset.
    :param ticker: The stock ticker (e.g. 'IAG'). If set to 'all' or omitted, returns global dataset metadata.
    :param date: Date formatted as YYYY-MM-DD. If set to 'all' or omitted, returns global dataset metadata.
    """
    # 1. Check if the model is requesting general dataset dimensions/metadata
    if (not ticker or ticker.lower().strip() in ["all", "history", "metadata", "none", "null"]) or \
       (not date or date.lower().strip() in ["all", "history", "metadata", "none", "null"]):
        
        if not os.path.exists(CLEANED_ASX_DIR):
            return f"Error: Cleaned ASX directory not found at {CLEANED_ASX_DIR}."
            
        try:
            # Find all partitioned ticker files
            target_files = glob.glob(os.path.join(CLEANED_ASX_DIR, "*.jsonl"))
            file_count = len(target_files)
            
            if file_count > 0:
                # Read a sample file (e.g., the first one) to determine row count and date ranges
                sample_file = target_files[0]
                rows = []
                with open(sample_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            rows.append(json.loads(line))
                
                row_count = len(rows)
                if row_count > 0:
                    min_date = rows[0].get("date")
                    max_date = rows[-1].get("date")
                    
                    # Return standard schema structure for the synthesizer
                    return json.dumps({
                        "ticker_files_count": file_count,
                        "rows_per_file": row_count,
                        "min_date": min_date,
                        "max_date": max_date,
                        "tickers_present": [os.path.basename(f).replace(".jsonl", "") for f in target_files]
                    })
            return f"No ASX ticker data found in {CLEANED_ASX_DIR}."
        except Exception as e:
            return f"Error reading ASX metadata: {str(e)}"

    # 2. Otherwise, run standard point-lookup query
    clean_ticker = ticker.strip().replace(".AX", "").replace(".ax", "").upper()
    target_date = date.strip()
    ticker_file_path = os.path.join(CLEANED_ASX_DIR, f"{clean_ticker}.jsonl")
    
    if not os.path.exists(ticker_file_path):
        return f"No ASX record found. Ticker '{clean_ticker}' does not exist in local dataset."

    try:
        with open(ticker_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("date") == target_date:
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

def query_afr_news(query: str, date_filter: str = None, max_results: int = 3) -> str:
    """Scans the cleaned local AFR news corpus folder for articles."""
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
                    if date_filter and record.get("publication_date") != date_filter.strip():
                        continue

                    headline = record.get("headline", "").lower()
                    intro = record.get("intro", "").lower()
                    text = record.get("text", "").lower()

                    is_match = all(term in headline or term in intro or term in text for term in search_terms)
                    if is_match:
                        matches.append({
                            "headline": record["headline"],
                            "publication_date": record["publication_date"],
                            "intro": record["intro"],
                            "text": record["text"][:800] + "..." if len(record["text"]) > 800 else record["text"]
                        })
                        if len(matches) >= max_results:
                            break
        if matches:
            return json.dumps(matches)
        else:
            return f"No AFR news articles found matching search query '{query}'."
    except Exception as e:
        return f"Error searching AFR database: {str(e)}"


@traceable(run_type="tool", name="execute_tool")
def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    try:
        if "query_rba_cash_rate" in tool_name:
            return query_rba_cash_rate(arguments.get("effective_date"))
        elif "query_asx_prices" in tool_name:
            return query_asx_prices(arguments.get("ticker", ""), arguments.get("date", ""))
        elif "query_afr_news" in tool_name:
            return query_afr_news(arguments.get("query", ""), arguments.get("date_filter"))
        else:
            return f"Error: Tool '{tool_name}' not recognized."
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


# --- 3. Robust Graph Nodes ---

def reasoning_planner(state: AgentState) -> Dict[str, Any]:
    """Qwen-35B acts as the planner with robust error-handling wrappers."""
    question = state["question"]
    context = state["context"]
    loop_count = state.get("loop_count", 0)

    system_prompt = (
    "You are an expert financial planning assistant. You have access to three local database tools:\n\n"
    "1. query_rba_cash_rate(effective_date: str) -> Expects YYYY-MM-DD. "
    "   IMPORTANT: If a question requires counting, scanning, or identifying historical trends across "
    "   the entire dataset, pass effective_date=all to retrieve the full history.\n"
    "2. query_asx_prices(ticker: str, date: str) -> Expects clean ticker (e.g., 'AGL') and date as YYYY-MM-DD. "
    "   IMPORTANT: If a question asks about dataset metadata, dimensions, or file statistics across the "
    "   entire ASX dataset, pass ticker=all and date=all to retrieve general file counts, row counts, and date ranges.\n"
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
    
    "Example 2: Querying ASX Dataset Dimensions & Metadata (Easy/Metadata)\n"
    "User Question: What are the dimensions and common date range of the ASX dataset?\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The user wants to know general dataset parameters (number of ticker files, row counts, and the date range) of the entire ASX dataset.\n"
    "2. Since this query targets directory and file statistics rather than a single stock price, I must request global ASX dataset metadata.\n"
    "3. I will call query_asx_prices with ticker=all and date=all.\n"
    "</thinking>\n"
    "CALL: query_asx_prices|ticker=all,date=all\n\n"
    
    "Example 3: Analyzing a specific temporal range (Medium/Cycle)\n"
    "User Question: Across the 2011-2013 easing period, how many cuts occurred and how far did the target fall?\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The user wants to analyze a specific temporal window (2011-2013 easing period).\n"
    "2. To count the cuts and identify the start/end rate targets over this period, I must inspect the historical sequence of records.\n"
    "3. Retrieving the complete RBA dataset is the most reliable way to let the synthesis engine filter, count, and sum target movements between 2011 and 2013.\n"
    "4. I will call the RBA tool with effective_date=all.\n"
    "</thinking>\n"
    "CALL: query_rba_cash_rate|effective_date=all\n\n"
    
    "Example 4: Querying the AFR news on a specific event\n"
    "User Question: What were the key points of the coroner's decision regarding the Downer EDI inquest mentioned in early 2015?\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The user is asking about a specific news event involving Downer EDI and a coroner's decision in early 2015.\n"
    "2. I need to search the AFR news corpus for relevant articles using keywords like 'Downer EDI' and 'coroner' or 'Alec Meikle'.\n"
    "3. I will call the query_afr_news tool with an appropriate query string.\n"
    "</thinking>\n"
    "CALL: query_afr_news|query=Downer EDI coroner\n\n"
)

    user_message = f"User Question: {question}\n\nAccumulated Data Context:\n{context}\n\nDetermine the next step."

    try:
        response_text = call_llm(BRAIN_MODEL, system_prompt, user_message)
    except Exception as e:
        logger.error("Exception occurred inside reasoning_planner: %s", str(e))
        # Gracefully handle planner failure by triggering fallback synthesis
        return {
            "plan": "DECISION: READY",
            "tool_calls": [],
            "error_message": f"Planner failure: {str(e)}",
            "loop_count": loop_count + 1
        }

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
        except Exception as parser_err:
            logger.warning("Failed to parse tool call: %s", str(parser_err))

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

    try:
        for call in tool_calls:
            output = execute_tool(call["name"], call["arguments"])
            outputs.append(output)
            
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
            "tool_calls": [],
            "tool_trace": current_trace + new_trace_entries
        }
    except Exception as e:
        logger.error("Exception occurred inside execute_tools_node: %s", str(e))
        # Log failure inside context and break the execution loop
        error_context = f"{current_context}\n[System Exception during tool execution: {str(e)}]".strip()
        return {
            "context": error_context,
            "tool_calls": [],
            "error_message": f"Tool Execution failure: {str(e)}"
        }


def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """The fine-tuned model synthesizes final answer, handles partial error context gracefully."""
    question = state["question"]
    context = state["context"]
    error_msg = state.get("error_message", "")

    if DOMAIN_PREDICT_MODE == "mock":
        return {"final_answer": f"[mock] Context gathered for: {question}\n{context}".strip()}

    # If an error occurred, append standard fallback instruction to keep output grounded
    system_prompt = (
        "You are an expert financial analysis synthesizer. Generate a direct, grounded answer "
        "to the question based strictly on the provided context (which is formatted as JSON blocks)."
    )
    if error_msg:
        system_prompt += (
            f"\n\nNOTE: An unexpected pipeline error occurred during execution: {error_msg}. "
            "Please synthesize the best possible answer utilizing only the partial context available."
        )

    user_message = f"Context Blocks:\n{context}\n\nQuestion: {question}"

    try:
        answer = call_llm(DOMAIN_FT_MODEL, system_prompt, user_message)
    except Exception as e:
        logger.critical("Catastrophic synthesizer failure: %s", str(e))
        # Absolute final fallback if Nemotron cannot be reached
        answer = f"Error: Unable to synthesize answer. Partial Context: {context[:500]}..."

    return {"final_answer": answer}


# --- 4. Conditional Routing (Enhanced with Error-Bypass) ---
def router(state: AgentState) -> Literal["execute_tools", "synthesize_answer"]:
    # 1. If an error message has been recorded, bypass standard iterations and finalize immediately
    if state.get("error_message"):
        return "synthesize_answer"

    # 2. Enforce execution loop safety limits
    if state["loop_count"] >= state["max_loops"]:
        return "synthesize_answer"
    
    # 3. Standard parsing transitions
    plan = state.get("plan", "")
    if "DECISION: READY" in plan or not state.get("tool_calls"):
        return "synthesize_answer"
        
    return "execute_tools"


# --- 5. Compile the Graph ---
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


# --- 6. Execution Endpoint ---
@traceable(
    run_type="chain",
    name="financial_agent",
    metadata={
        "brain_model": BRAIN_MODEL,
        "domain_ft_model": DOMAIN_FT_MODEL,
        "domain_predict_mode": DOMAIN_PREDICT_MODE,
    },
)
def run_financial_agent(question: str, max_loops: int = MAX_AGENT_STEPS) -> Dict[str, Any]:
    initial_state = {
        "question": question,
        "plan": "",
        "tool_calls": [],
        "tool_outputs": [],
        "context": "",
        "final_answer": "",
        "loop_count": 0,
        "max_loops": max_loops,
        "tool_trace": [],
        "error_message": ""  # Initialize empty error state
    }
    
    final_state = app.invoke(initial_state)
    
    return {
        "answer": final_state.get("final_answer", ""),
        "steps": final_state.get("loop_count", 0),
        "tool_trace": final_state.get("tool_trace", [])
    }