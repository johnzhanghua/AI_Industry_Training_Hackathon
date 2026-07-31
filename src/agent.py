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

# def query_asx_prices(ticker: str = None, date: str = None) -> str:
#     """
#     Queries the partitioned local cleaned ASX dataset.
#     :param ticker: The stock ticker (e.g. 'IAG'). If set to 'all' or omitted, returns global dataset metadata.
#     :param date: Date formatted as YYYY-MM-DD. If set to 'all' or omitted, returns global dataset metadata.
#     """
#     # 1. Check if the model is requesting general dataset dimensions/metadata
#     if (not ticker or ticker.lower().strip() in ["all", "history", "metadata", "none", "null"]) or \
#        (not date or date.lower().strip() in ["all", "history", "metadata", "none", "null"]):
        
#         if not os.path.exists(CLEANED_ASX_DIR):
#             return f"Error: Cleaned ASX directory not found at {CLEANED_ASX_DIR}."
            
#         try:
#             # Find all partitioned ticker files
#             target_files = glob.glob(os.path.join(CLEANED_ASX_DIR, "*.jsonl"))
#             file_count = len(target_files)
            
#             if file_count > 0:
#                 # Read a sample file (e.g., the first one) to determine row count and date ranges
#                 sample_file = target_files[0]
#                 rows = []
#                 with open(sample_file, 'r', encoding='utf-8') as f:
#                     for line in f:
#                         if line.strip():
#                             rows.append(json.loads(line))
                
#                 row_count = len(rows)
#                 if row_count > 0:
#                     min_date = rows[0].get("date")
#                     max_date = rows[-1].get("date")
                    
#                     # Return standard schema structure for the synthesizer
#                     return json.dumps({
#                         "ticker_files_count": file_count,
#                         "rows_per_file": row_count,
#                         "min_date": min_date,
#                         "max_date": max_date,
#                         "tickers_present": [os.path.basename(f).replace(".jsonl", "") for f in target_files]
#                     })
#             return f"No ASX ticker data found in {CLEANED_ASX_DIR}."
#         except Exception as e:
#             return f"Error reading ASX metadata: {str(e)}"

#     # 2. Otherwise, run standard point-lookup query
#     clean_ticker = ticker.strip().replace(".AX", "").replace(".ax", "").upper()
#     target_date = date.strip()
#     ticker_file_path = os.path.join(CLEANED_ASX_DIR, f"{clean_ticker}.jsonl")
    
#     if not os.path.exists(ticker_file_path):
#         return f"No ASX record found. Ticker '{clean_ticker}' does not exist in local dataset."

#     try:
#         with open(ticker_file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 if not line.strip():
#                     continue
#                 record = json.loads(line)
#                 if record.get("date") == target_date:
#                     return json.dumps({
#                         "ticker": record["ticker"],
#                         "date": record["date"],
#                         "open": record["open"],
#                         "high": record["high"],
#                         "low": record["low"],
#                         "close": record["close"],
#                         "volume": record["volume"]
#                     })
#         return f"No stock prices found for ticker '{clean_ticker}' on date {target_date}."
#     except Exception as e:
#         return f"Error reading ASX record for {clean_ticker}: {str(e)}"

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

def query_asx_prices(ticker: str = None, date: str = None) -> str:
    """
    Queries the partitioned local cleaned ASX dataset.
    :param ticker: Cleaned stock ticker (e.g. 'IAG'), a comma-separated list (e.g. 'AGL, IAG'), 
                   or set to 'all' to retrieve data for all available tickers.
    :param date: Specific date formatted as YYYY-MM-DD (e.g. '2015-01-02'), 
                 or a 4-digit year (e.g. '2018') to retrieve start/end records of that year.
    """
    if not os.path.exists(CLEANED_ASX_DIR):
        return f"Error: Cleaned ASX directory not found at {CLEANED_ASX_DIR}."

    ticker_str = (ticker or "").strip()
    date_str = (date or "").strip()

    # Helper to get the first and last trading record of a given year for a single file
    def get_year_bounds(file_path: str, year: str) -> dict:
        first_row = None
        last_row = None
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("date", "").startswith(f"{year}-"):
                        if first_row is None:
                            first_row = record
                        last_row = record
            if first_row and last_row:
                return {
                    "ticker": first_row["ticker"],
                    "start_date": first_row["date"],
                    "start_close": first_row["close"],
                    "end_date": last_row["date"],
                    "end_close": last_row["close"]
                }
        except Exception:
            pass
        return None

    # Helper to get exact date record from a single file
    def get_date_record(file_path: str, target_date: str) -> dict:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("date") == target_date:
                        return record
        except Exception:
            pass
        return None

    # CASE 1: Query is requesting historical bounds for a given Year (e.g., "2018")
    if len(date_str) == 4 and date_str.isdigit():
        target_year = date_str
        
        # Subcase A: All tickers
        if not ticker_str or ticker_str.lower() in ["all", "history", "metadata", "none", "null"]:
            target_files = glob.glob(os.path.join(CLEANED_ASX_DIR, "*.jsonl"))
            results = []
            for file_path in target_files:
                bounds = get_year_bounds(file_path, target_year)
                if bounds:
                    results.append(bounds)
            return json.dumps(results)
            
        # Subcase B: Specific comma-separated list of tickers
        else:
            tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
            results = []
            for t in tickers:
                file_path = os.path.join(CLEANED_ASX_DIR, f"{t}.jsonl")
                if os.path.exists(file_path):
                    bounds = get_year_bounds(file_path, target_year)
                    if bounds:
                        results.append(bounds)
            return json.dumps(results) if results else f"No records found for tickers {ticker_str} in {target_year}."

    # CASE 2: Query is requesting general folder metadata (no specific parameters)
    if (not ticker_str or ticker_str.lower() in ["all", "history", "metadata", "none", "null"]) and \
       (not date_str or date_str.lower() in ["all", "history", "metadata", "none", "null"]):
        
        target_files = glob.glob(os.path.join(CLEANED_ASX_DIR, "*.jsonl"))
        file_count = len(target_files)
        if file_count > 0:
            sample_file = target_files[0]
            rows = []
            with open(sample_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        rows.append(json.loads(line))
            row_count = len(rows)
            if row_count > 0:
                return json.dumps({
                    "ticker_files_count": file_count,
                    "rows_per_file": row_count,
                    "min_date": rows[0].get("date"),
                    "max_date": rows[-1].get("date"),
                    "tickers_present": [os.path.basename(f).replace(".jsonl", "") for f in target_files]
                })
        return f"No ASX ticker data found in {CLEANED_ASX_DIR}."

    # CASE 3: Query is a point-lookup for a specific YYYY-MM-DD date
    if date_str:
        # Subcase A: All tickers on this date
        if not ticker_str or ticker_str.lower() in ["all", "history", "none"]:
            target_files = glob.glob(os.path.join(CLEANED_ASX_DIR, "*.jsonl"))
            results = []
            for file_path in target_files:
                rec = get_date_record(file_path, date_str)
                if rec:
                    results.append(rec)
            return json.dumps(results) if results else f"No records found for date {date_str}."
        
        # Subcase B: Commas-separated list of tickers on this date
        else:
            tickers = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
            results = []
            for t in tickers:
                file_path = os.path.join(CLEANED_ASX_DIR, f"{t}.jsonl")
                if os.path.exists(file_path):
                    rec = get_date_record(file_path, date_str)
                    if rec:
                        results.append(rec)
            return json.dumps(results) if results else f"No records found for tickers {ticker_str} on date {date_str}."

    return "Invalid parameters passed to query_asx_prices."

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
    "   If the question requires aggregate counts or trends, and the RBA history is NOT already present "
    "   in the context, pass effective_date=all to retrieve it.\n"
    "2. query_asx_prices(ticker: str, date: str) -> Expects clean ticker (e.g., 'AGL'), a comma-separated list of tickers "
    "   (e.g., 'AGL, IAG'), or 'all'. Expects date as YYYY-MM-DD (e.g. '2015-01-02') or a 4-digit year (e.g. '2018').\n"
    "   - If a question asks about dataset metadata, pass ticker=all and date=all.\n"
    "   - If a question requires comparing stock returns or pricing across a whole year, pass ticker=all and date=YYYY "
    "     (e.g. date=2018) to retrieve the start and end prices of that year for all tickers.\n"
    "3. query_afr_news(query: str, date_filter: str) -> Searches news, optional date as YYYY-MM-DD\n\n"
    
    "CRITICAL CONSTRAINTS & INSTRUCTIONS:\n"
    "- Your response must start immediately with the '<thinking>' tag. Do not write any preambles, introductory sentences, or plaintext thoughts before the <thinking> tag.\n"
    "- CONTEXT-FIRST RULE: Always inspect the 'Accumulated Data Context' first. If the context already contains the data, numbers, or metadata needed to fully answer the user's question, do NOT call any tools. Proceed immediately to 'DECISION: READY'.\n"
    "- Following the </thinking> tag, you must output exactly one action line: either a tool CALL or a final DECISION.\n"
    "- Do not write any conversational filler outside the <thinking> block.\n\n"
    
    "EXAMPLES:\n\n"
    
    "Example 1: Analyzing the entire history (No Context present -> Call Tool)\n"
    "User Question: From the first RBA record to the last, how many cash-rate decisions changed the rate?\n"
    "Accumulated Data Context:\n"
    "None\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The user is asking for aggregate statistics (counts of changes, increases, and decreases) over the entire historical range of the RBA dataset.\n"
    "2. The RBA history is not present in the current context.\n"
    "3. I must call the tool with effective_date=all.\n"
    "</thinking>\n"
    "CALL: query_rba_cash_rate|effective_date=all\n\n"
    
    "Example 2: Analyzing Annual Ticker Performance (No Context present -> Call Tool)\n"
    "User Question: Excluding Tabcorp, which ticker had the best and worst 2018 return?\n"
    "Accumulated Data Context:\n"
    "None\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The user wants to compare the 2018 annual returns across all tickers (excluding Tabcorp).\n"
    "2. To calculate 2018 returns, I need the starting and ending prices for 2018 for all tickers.\n"
    "3. Instead of querying each ticker individually, I can retrieve the 2018 start and end bounds for all tickers by calling query_asx_prices with ticker=all and date=2018.\n"
    "4. I will make this call.\n"
    "</thinking>\n"
    "CALL: query_asx_prices|ticker=all,date=2018\n\n"
    
    "Example 3: Querying ASX Dataset Metadata (No Context present -> Call Tool)\n"
    "User Question: What are the dimensions and common date range of the ASX dataset?\n"
    "Accumulated Data Context:\n"
    "None\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The user wants to know general dataset metadata and dimensions for the ASX dataset.\n"
    "2. This metadata is not present in the current context.\n"
    "3. I must call query_asx_prices with ticker=all and date=all to retrieve the dimensions.\n"
    "</thinking>\n"
    "CALL: query_asx_prices|ticker=all,date=all\n\n"
    
    "Example 4: Ready to Synthesize (Context is already present -> Finish)\n"
    "User Question: Excluding Tabcorp, which ticker had the best and worst 2018 return?\n"
    "Accumulated Data Context:\n"
    "[{\"ticker\": \"AGL\", \"start_date\": \"2018-01-02\", \"start_close\": 22.10, \"end_date\": \"2018-12-31\", \"end_close\": 20.30}, ...]\n"
    "Assistant Output:\n"
    "<thinking>\n"
    "1. The accumulated context contains the starting and ending 2018 prices for all stock tickers.\n"
    "2. The necessary parameters to calculate annual percentage returns are available.\n"
    "3. No further queries are required. I will transition to the analysis and synthesis phase.\n"
    "</thinking>\n"
    "DECISION: READY"
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

def analytical_processor(state: AgentState) -> Dict[str, Any]:
    """
    NEW NODE: Qwen-35B (BRAIN_MODEL) receives the user query and the accumulated
    context. It performs the rigorous mathematical, chronological, and logical analysis.
    """
    question = state["question"]
    context = state["context"]
    error_msg = state.get("error_message", "")

    system_prompt = (
        "You are an expert financial market analyst. Your job is to perform a detailed, "
        "step-by-step mathematical, statistical, or logical analysis of the provided data context "
        "to answer the user's query.\n"
        "Provide a highly accurate, calculated analysis report. Show your math and chronological breakdowns."
    )
    if error_msg:
        system_prompt += f"\n\nNOTE: System encountered errors prior to this step: {error_msg}."

    user_message = f"Raw Context Blocks:\n{context}\n\nUser Question: {question}"

    try:
        analysis_report = call_llm(BRAIN_MODEL, system_prompt, user_message)
    except Exception as e:
        logger.error("Exception in analytical_processor: %s", str(e))
        analysis_report = f"Analytical processing execution failed: {str(e)}"

    return {"analysis_report": analysis_report}

# --- 4. Conditional Routing (Enhanced with Error-Bypass) ---
# def router(state: AgentState) -> Literal["execute_tools", "synthesize_answer"]:
#     # 1. If an error message has been recorded, bypass standard iterations and finalize immediately
#     if state.get("error_message"):
#         return "synthesize_answer"

#     # 2. Enforce execution loop safety limits
#     if state["loop_count"] >= state["max_loops"]:
#         return "synthesize_answer"
    
#     # 3. Standard parsing transitions
#     plan = state.get("plan", "")
#     if "DECISION: READY" in plan or not state.get("tool_calls"):
#         return "synthesize_answer"
        
#     return "execute_tools"

# --- 4. Conditional Routing (Enhanced with Error-Bypass) ---
def router(state: AgentState) -> Literal["execute_tools", "analyze_data"]:
    # 1. If an error message has been recorded, bypass standard iterations and analyze immediately
    if state.get("error_message"):
        return "analyze_data"

    # 2. Enforce execution loop safety limits
    if state["loop_count"] >= state["max_loops"]:
        return "analyze_data"
    
    # 3. Standard parsing transitions
    plan = state.get("plan", "")
    if "DECISION: READY" in plan or not state.get("tool_calls"):
        return "analyze_data"
        
    return "execute_tools"

# --- 5. Compile the Graph ---
# workflow = StateGraph(AgentState)

# workflow.add_node("reasoning_planner", reasoning_planner)
# workflow.add_node("execute_tools", execute_tools_node)
# workflow.add_node("synthesize_answer", synthesis_node)

# workflow.add_edge(START, "reasoning_planner")
# workflow.add_conditional_edges(
#     "reasoning_planner",
#     router,
#     {
#         "execute_tools": "execute_tools",
#         "synthesize_answer": "synthesize_answer"
#     }
# )
# workflow.add_edge("execute_tools", "reasoning_planner")
# workflow.add_edge("synthesize_answer", END)

# app = workflow.compile()
workflow = StateGraph(AgentState)

workflow.add_node("reasoning_planner", reasoning_planner)
workflow.add_node("execute_tools", execute_tools_node)
workflow.add_node("analyze_data", analytical_processor)  # Add new analytical node
workflow.add_node("synthesize_answer", synthesis_node)

workflow.add_edge(START, "reasoning_planner")
workflow.add_conditional_edges(
    "reasoning_planner",
    router,
    {
        "execute_tools": "execute_tools",
        "analyze_data": "analyze_data"  # Route to analyzer instead of straight to synthesis
    }
)
workflow.add_edge("execute_tools", "reasoning_planner")
workflow.add_edge("analyze_data", "synthesize_answer")  # Analyzer flows to synthesizer
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