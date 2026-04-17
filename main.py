# main.py
# Part H: Complete System with All APIs

from dotenv import load_dotenv
import os
import uuid

# Core components
from memory.memory_manager import MemoryManager
from retrieval.rag_pipeline import RAGPipeline

# Agents
from agents.profile_agent import ProfileAgent
from agents.research_agent import ResearchAgent
from agents.advisor_agent import AdvisorAgent

# LangGraph workflow
from graph.workflow import UniversityFinderWorkflow

# MCP Servers
from tools.filesystem_mcp import FilesystemMCPServer
from tools.database_mcp import DatabaseMCPServer

# External APIs
from tools.search_tool import TavilySearchTool
from tools.currency_tool import CurrencyTool

load_dotenv()

# ─────────────────────────────────────────
# INITIALIZE ALL COMPONENTS
# ─────────────────────────────────────────

print("\n" + "="*50)
print("   🎓 International University Finder AI")
print("      Complete Multi-Agent System v4.0")
print("="*50 + "\n")

# Memory
memory = MemoryManager()

# RAG
rag = RAGPipeline()

# External APIs
tavily   = TavilySearchTool()
currency = CurrencyTool()

# Agents — pass Tavily to Research Agent
profile_agent  = ProfileAgent()
research_agent = ResearchAgent(
    rag_pipeline = rag,
    search_tool  = tavily    # NEW: live search
)
advisor_agent  = AdvisorAgent()

# LangGraph workflow
workflow = UniversityFinderWorkflow(
    profile_agent  = profile_agent,
    research_agent = research_agent,
    advisor_agent  = advisor_agent,
    memory_manager = memory,
)

# MCP Servers
filesystem_mcp = FilesystemMCPServer(base_path="outputs")
database_mcp   = DatabaseMCPServer(
    db_path="data/university_finder.db"
)

# Session ID
SESSION_ID = str(uuid.uuid4())[:8]

print(f"\n🔑 Session ID:    {SESSION_ID}")
print(f"📁 Reports:       {filesystem_mcp.get_outputs_path()}")
print(f"🗄️  Database:      {database_mcp.get_db_path()}")
print(f"🌐 Live Search:   {'✅ Active' if tavily.enabled else '❌ Disabled'}")
print(f"💱 Currency:      {'✅ Active' if currency.enabled else '❌ Disabled'}")
print("\n✅ All systems ready!\n")

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def print_help():
    print("""
─────────────────────────────────────────
📌 COMMANDS:
  'profile'    → show your profile
  'summary'    → show conversation summary
  'report'     → generate full report + save
  'convert'    → convert USD fees to PKR
  'files'      → list saved reports
  'history'    → show search history
  'stats'      → database statistics
  'graph'      → workflow diagram
  'clear'      → start fresh
  'quit'       → exit
─────────────────────────────────────────
    """)

def convert_and_display(amount_usd: float,
                         label: str = "Amount"):
    """Helper to convert and display USD to PKR."""
    result = currency.convert(amount_usd, "USD", "PKR")
    if result["success"]:
        print(f"  {label}: ${amount_usd:,} USD "
              f"= {result['formatted']}")
    else:
        print(f"  {label}: ${amount_usd:,} USD")

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────

def main():
    
    print("Type 'help' for commands\n")
    print("─" * 42)
    
    message_count = 0
    
    # Welcome returning students
    existing_profile = memory.get_profile()
    if existing_profile:
        completeness = profile_agent.assess_profile_completeness(
            existing_profile
        )
        pct = completeness["completeness_percent"]
        print(f"👋 Welcome back! Profile loaded ({pct}% complete)\n")
        database_mcp.save_student_profile(
            existing_profile, SESSION_ID
        )
    
    while True:
        user_input = input("You: ").strip()
        
        # ── COMMANDS ───────────────────────────────
        
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n🎓 Goodbye! Best of luck!")
            stats = database_mcp.get_stats()
            if stats["success"]:
                print(f"\n📊 Session Stats:")
                print(f"   Searches:        {stats['searches_logged']}")
                print(f"   Recommendations: {stats['recommendations']}")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "help":
            print_help()
            continue
        
        if user_input.lower() == "profile":
            print(memory.get_profile_summary())
            continue
        
        if user_input.lower() == "summary":
            print(f"\n📝 SUMMARY:\n{memory.get_summary()}\n")
            continue
        
        if user_input.lower() == "graph":
            workflow.visualize()
            continue
        
        if user_input.lower() == "clear":
            memory.clear_all_memory()
            message_count = 0
            print("✅ Cleared!\n")
            continue
        
        # ── CURRENCY CONVERT COMMAND ────────────────
        if user_input.lower() == "convert":
            print("\n💱 CURRENCY CONVERTER (USD → PKR)\n")
            
            # Show costs for common amounts
            amounts = [10000, 13000, 15000, 18000, 25000]
            for amt in amounts:
                convert_and_display(amt, f"${amt:,}/year tuition")
            
            # Also show profile budget if available
            profile = memory.get_profile()
            if profile.get("budget_tuition_usd"):
                budget = float(profile["budget_tuition_usd"])
                print(f"\n  Your budget: ", end="")
                convert_and_display(budget, "Budget")
            print()
            continue
        
        # ── FILES COMMAND ───────────────────────────
        if user_input.lower() == "files":
            result = filesystem_mcp.list_reports()
            if result["count"] == 0:
                print("\n📁 No saved reports yet.\n")
            else:
                print(f"\n📁 SAVED REPORTS ({result['count']}):")
                print("─" * 40)
                for f in result["files"]:
                    print(f"  📄 {f['filename']}")
                    print(f"     {f['size_kb']} KB | "
                          f"{f['saved_at']}")
                print()
            continue
        
        # ── HISTORY COMMAND ─────────────────────────
        if user_input.lower() == "history":
            result = database_mcp.get_search_history(limit=10)
            if result["count"] == 0:
                print("\n📊 No search history yet.\n")
            else:
                print(f"\n📊 SEARCH HISTORY:")
                print("─" * 40)
                for h in result["history"]:
                    print(f"  🔍 {h['query'][:45]}...")
                    print(f"     {h['searched_at']}")
                print()
            continue
        
        # ── STATS COMMAND ───────────────────────────
        if user_input.lower() == "stats":
            result = database_mcp.get_stats()
            if result["success"]:
                print(f"""
📊 DATABASE STATISTICS:
─────────────────────────────
  👤 Profiles saved:   {result['profiles_saved']}
  🔍 Searches logged:  {result['searches_logged']}
  🎯 Recommendations:  {result['recommendations']}
─────────────────────────────
""")
            continue
        
        # ── REPORT COMMAND ──────────────────────────
        if user_input.lower() == "report":
            profile = memory.get_profile()
            if not profile:
                print("\n⚠️ No profile yet!\n")
                continue
            
            print("\n🎯 Generating full report...\n")
            
            # Research
            research_results = research_agent.research_by_profile(
                profile, memory.get_chat_history()
            )
            
            # Advise
            report = advisor_agent.generate_full_report(
                profile, research_results,
                memory.get_chat_history()
            )
            
            # ── Add currency conversions to report ──
            currency_section = "\n\n💱 FEES IN PKR:\n" + "─"*30 + "\n"
            
            university_fees = {
                "University of North Texas":    13000,
                "University of Texas Arlington": 15000,
                "University of Houston":         16000,
                "Texas A&M":                     17000,
                "University of Texas Dallas":    18000,
                "TU Berlin":                     0,
                "University of Hamburg":         0,
                "University of Amsterdam":       18000,
            }
            
            for uni, fee in university_fees.items():
                if fee == 0:
                    currency_section += \
                        f"  {uni}: FREE (public university)\n"
                else:
                    result = currency.convert(fee, "USD", "PKR")
                    if result["success"]:
                        currency_section += (
                            f"  {uni}: ${fee:,}/yr = "
                            f"{result['formatted']}/yr\n"
                        )
            
            full_report = report + currency_section
            
            print(f"\n{'='*50}")
            print("📊 UNIVERSITY RECOMMENDATION REPORT")
            print('='*50)
            print(full_report)
            print('='*50 + "\n")
            
            # Save to file (Filesystem MCP)
            student_name = profile.get("name", "student")
            save_result  = filesystem_mcp.save_report(
                full_report, student_name
            )
            if save_result["success"]:
                print(f"💾 Saved: {save_result['filename']}")
            
            # Save profile file
            filesystem_mcp.save_profile(profile, student_name)
            
            # Save to database (Database MCP)
            database_mcp.save_recommendation(
                SESSION_ID,
                list(profile.get("target_countries", [])),
                full_report
            )
            print("🗄️  Saved to database\n")
            continue
        
        # ── RUN LANGGRAPH WORKFLOW ──────────────────
        
        print("\n⚙️  Processing...\n")
        
        result   = workflow.run(user_input)
        ai_reply = result.get("final_response", "")
        
        if not ai_reply:
            ai_reply = "I couldn't process that. Try again."
        
        # Show agent label
        current_node = result.get("current_node", "unknown")
        node_labels  = {
            "profile":  "🧑 Profile Agent",
            "research": "🔍 Research Agent",
            "advisor":  "🎯 Advisor Agent",
            "router":   "🔀 Router",
        }
        node_label = node_labels.get(current_node, current_node)
        
        print(f"[{node_label}]")
        print(f"AI: {ai_reply}\n")
        
        # Save to memory
        memory.save_message(user_input, ai_reply)
        message_count += 1
        
        # Log search to database (Database MCP)
        if current_node in ["research", "advisor"]:
            profile = memory.get_profile()
            database_mcp.log_search(
                session_id    = SESSION_ID,
                query         = user_input,
                results_count = 5,
                countries     = str(profile.get(
                    "target_countries", ""
                )),
                field         = profile.get(
                    "target_field", ""
                ),
            )
        
        # Update every 4 messages
        if message_count % 4 == 0:
            print("🔄 Updating profile...\n")
            updated = memory.extract_and_save_profile(
                memory.get_chat_history()
            )
            memory.update_summary()
            if updated:
                database_mcp.save_student_profile(
                    updated, SESSION_ID
                )

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    main()