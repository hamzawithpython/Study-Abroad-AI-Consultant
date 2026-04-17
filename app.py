# app.py
# Gradio UI for International University Finder AI
# Run with: python app.py

import gradio as gr
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

print("\n🚀 Starting International University Finder AI...\n")

memory = MemoryManager(fresh_start=True)
rag            = RAGPipeline()
tavily         = TavilySearchTool()
currency       = CurrencyTool()
profile_agent  = ProfileAgent()
research_agent = ResearchAgent(rag_pipeline=rag, search_tool=tavily)
advisor_agent  = AdvisorAgent()
workflow       = UniversityFinderWorkflow(
    profile_agent  = profile_agent,
    research_agent = research_agent,
    advisor_agent  = advisor_agent,
    memory_manager = memory,
)
filesystem_mcp = FilesystemMCPServer(base_path="outputs")
database_mcp   = DatabaseMCPServer(
    db_path="data/university_finder.db"
)

SESSION_ID = str(uuid.uuid4())[:8]
print(f"\n✅ All systems ready! Session: {SESSION_ID}\n")

# ─────────────────────────────────────────
# LOGGING SYSTEM
# ─────────────────────────────────────────

from datetime import datetime
from collections import deque

# Store last 50 log entries
log_entries = deque(maxlen=50)

def add_log(message: str, level: str = "info"):
    """
    Adds a log entry with timestamp.
    
    Args:
        message: log message
        level: info, success, warning, search, agent
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    icons = {
        "info":    "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "search":  "🔍",
        "agent":   "🤖",
        "memory":  "💾",
        "api":     "🌐",
        "router":  "🔀",
        "profile": "🧑",
        "research":"🔍",
        "advisor": "🎯",
        "mcp":     "🔌",
        "currency":"💱",
        "error":   "❌",
    }
    
    icon  = icons.get(level, "•")
    entry = f"[{timestamp}] {icon} {message}"
    log_entries.append(entry)

def get_log_text():
    """Returns all log entries as formatted string."""
    if not log_entries:
        return "No activity yet. Start a conversation!"
    return "\n".join(reversed(list(log_entries)))

# Log startup
add_log(f"Session started: {SESSION_ID}", "success")
add_log(f"Live Search: {'Active' if tavily.enabled else 'Disabled'}", "info")
add_log(f"Currency API: {'Active' if currency.enabled else 'Disabled'}", "info")
add_log("All systems ready!", "success")

# ─────────────────────────────────────────
# CORE CHAT FUNCTION
# ─────────────────────────────────────────

def chat(user_message, history):
    """
    Main chat function called by Gradio.
    """
    
    if not user_message.strip():
        return ""
    
    # Log user message
    short_msg = user_message[:40] + "..." \
                if len(user_message) > 40 else user_message
    add_log(f"User: {short_msg}", "info")
    add_log("Router analyzing message...", "router")
    
    # Run LangGraph workflow
    result   = workflow.run(user_message)
    ai_reply = result.get("final_response", "")
    
    if not ai_reply:
        ai_reply = "I couldn't process that. Please try again."
    
    # Show which agent responded
    current_node = result.get("current_node", "unknown")
    node_labels  = {
        "profile":  "🧑 Profile Agent",
        "research": "🔍 Research Agent",
        "advisor":  "🎯 Advisor Agent",
    }
    node_label = node_labels.get(current_node, "🤖 AI")
    
    # Log which agent handled it
    add_log(f"Routed to: {node_label}", "router")
    add_log(f"{node_label} responding...", "agent")
    
    # Log RAG search if research agent
    if current_node == "research":
        profile = memory.get_profile()
        field   = profile.get("target_field", "")
        countries = profile.get("target_countries", "")
        add_log(f"RAG search: {field} {countries}", "search")
        if tavily.enabled:
            add_log("Tavily live search running...", "api")
    
    # Log advisor
    if current_node == "advisor":
        add_log("Generating recommendations...", "advisor")
        add_log("Matching profile vs requirements...", "advisor")
    
    # Save to memory
    memory.save_message(user_message, ai_reply)
    add_log("Conversation saved to memory", "memory")
    
    # Log search to database
    if current_node in ["research", "advisor"]:
        profile = memory.get_profile()
        database_mcp.log_search(
            session_id    = SESSION_ID,
            query         = user_message,
            results_count = 5,
            countries     = str(profile.get(
                "target_countries", ""
            )),
            field = profile.get("target_field", ""),
        )
        add_log("Search logged to database (MCP)", "mcp")
    
    add_log(f"Response sent ✓", "success")
    
    # Format response with agent label
    formatted_reply = f"**[{node_label}]**\n\n{ai_reply}"
    return formatted_reply

# ─────────────────────────────────────────
# ACTION FUNCTIONS (Button Handlers)
# ─────────────────────────────────────────

def get_profile():
    """Returns formatted student profile."""
    profile = memory.get_profile()
    if not profile:
        return "⚠️ No profile yet. Start a conversation first!"
    
    lines = ["## 📋 Your Student Profile\n"]
    lines.append("---")
    
    field_labels = {
        "bachelor_degree":    "🎓 Bachelor's Degree",
        "university":         "🏫 University",
        "gpa":                "📊 GPA",
        "gpa_scale":          "📊 GPA Scale",
        "graduation_year":    "📅 Graduation Year",
        "target_degree":      "🎯 Target Degree",
        "target_field":       "🔬 Target Field",
        "target_countries":   "🌍 Target Countries",
        "budget_tuition_usd": "💰 Tuition Budget (USD)",
        "budget_initial_pkr": "💵 Initial Budget (PKR)",
        "english_exemption":  "🗣️ English Exemption",
        "target_intake":      "📅 Target Intake",
        "career_goal":        "🚀 Career Goal",
    }
    
    for field, label in field_labels.items():
        value = profile.get(field)
        if value:
            lines.append(f"**{label}:** {value}")
    
    return "\n".join(lines)

def get_summary():
    """Returns conversation summary."""
    summary = memory.get_summary()
    if not summary or summary == "No conversation yet.":
        return "⚠️ No summary yet. Have a conversation first!"
    return f"## 📝 Conversation Summary\n\n{summary}"

def generate_report():
    """Generates and saves full recommendation report."""
    profile = memory.get_profile()
    if not profile:
        add_log("Report requested but no profile found", "warning")
        return "⚠️ No profile yet. Have a conversation first!"
    
    add_log("Full report generation started", "info")
    add_log("Research Agent searching universities...", "research")
    
    # Research
    research_results = research_agent.research_by_profile(
        profile, memory.get_chat_history()
    )
    add_log("Research complete", "success")
    add_log("Advisor Agent generating report...", "advisor")
    
    # Generate report
    report = advisor_agent.generate_full_report(
        profile, research_results, memory.get_chat_history()
    )
    
    # Add currency section
    currency_section = "\n\n---\n## 💱 Fees in PKR\n\n"
    university_fees  = {
        "University of North Texas":     13000,
        "University of Texas Arlington": 15000,
        "University of Houston":         16000,
        "Texas A&M":                     17000,
        "University of Texas Dallas":    18000,
        "University of Amsterdam":       18000,
        "TU Berlin":                     0,
        "University of Hamburg":         0,
    }
    
    for uni, fee in university_fees.items():
        if fee == 0:
            currency_section += f"- **{uni}:** FREE (public)\n"
        else:
            result = currency.convert(fee, "USD", "PKR")
            if result["success"]:
                currency_section += (
                    f"- **{uni}:** ${fee:,}/yr = "
                    f"{result['formatted']}/yr\n"
                )
    
    full_report = f"## 📊 University Recommendation Report\n\n" \
                  + report + currency_section
    
    # Save files via MCP
    add_log("Saving report to file (Filesystem MCP)...", "mcp")
    student_name = profile.get("name", "student")
    filesystem_mcp.save_report(report, student_name)
    filesystem_mcp.save_profile(profile, student_name)
    add_log("Report saved to file ✓", "success")
    
    add_log("Saving to database (Database MCP)...", "mcp")
    database_mcp.save_recommendation(
        SESSION_ID,
        list(profile.get("target_countries", [])),
        report
    )
    add_log("Saved to database ✓", "success")
    add_log("Currency conversion added to report", "currency")
    
    return full_report

def get_currency_info():
    """Returns currency conversion table."""
    lines = ["## 💱 USD to PKR Converter\n"]
    lines.append("| Tuition (USD) | Amount (PKR) |")
    lines.append("|---|---|")
    
    amounts = [5000, 10000, 12000, 13000, 15000, 
               16000, 17000, 18000, 20000, 25000]
    
    for amt in amounts:
        result = currency.convert(amt, "USD", "PKR")
        if result["success"]:
            lines.append(
                f"| ${amt:,}/year | {result['formatted']}/year |"
            )
        else:
            lines.append(f"| ${amt:,}/year | Unavailable |")
    
    # Add profile budget
    profile = memory.get_profile()
    if profile.get("budget_tuition_usd"):
        budget = float(profile["budget_tuition_usd"])
        result = currency.convert(budget, "USD", "PKR")
        if result["success"]:
            lines.append(f"\n**Your Budget:** ${budget:,} = "
                         f"{result['formatted']}")
    
    return "\n".join(lines)

def get_stats():
    """Returns database statistics."""
    stats  = database_mcp.get_stats()
    files  = filesystem_mcp.list_reports()
    
    lines = ["## 📊 System Statistics\n"]
    lines.append("### Database (MCP Server 2)")
    
    if stats["success"]:
        lines.append(f"- 👤 Profiles saved: **{stats['profiles_saved']}**")
        lines.append(f"- 🔍 Searches logged: **{stats['searches_logged']}**")
        lines.append(f"- 🎯 Recommendations: **{stats['recommendations']}**")
    
    lines.append("\n### Saved Files (MCP Server 1)")
    lines.append(f"- 📄 Reports saved: **{files['count']}**")
    
    if files["count"] > 0:
        lines.append("\n**Saved Reports:**")
        for f in files["files"]:
            lines.append(
                f"- 📄 `{f['filename']}` "
                f"({f['size_kb']} KB)"
            )
    
    lines.append(f"\n### Session Info")
    lines.append(f"- 🔑 Session ID: `{SESSION_ID}`")
    lines.append(
        f"- 🌐 Live Search: "
        f"{'✅ Active' if tavily.enabled else '❌ Off'}"
    )
    lines.append(
        f"- 💱 Currency: "
        f"{'✅ Active' if currency.enabled else '❌ Off'}"
    )
    
    return "\n".join(lines)

def clear_all():
    """Clears all memory and profile."""
    memory.clear_all_memory()
    return (
        "✅ Everything cleared! Start a fresh conversation.",
        [],
    )

# ─────────────────────────────────────────
# BUILD GRADIO INTERFACE
# ─────────────────────────────────────────

def build_interface():
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .chat-window {
        height: 500px !important;
    }
    .header-text {
        text-align: center;
        padding: 20px;
    }
    footer {display: none !important}
    """
    
    with gr.Blocks(
        title = "🎓 University Finder AI",
    ) as demo:
        
        # ── HEADER ─────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; padding:20px; 
                    background: linear-gradient(135deg, #1a237e, #283593);
                    border-radius: 12px; margin-bottom: 20px;">
            <h1 style="color:white; font-size:2em; margin:0;">
                🎓 International University Finder AI
            </h1>
            <p style="color:#90caf9; margin:8px 0 0 0; font-size:1.1em;">
                Your Personal Study Abroad Consultant
            </p>
            <p style="color:#64b5f6; margin:4px 0 0 0; font-size:0.9em;">
                Powered by LangChain • LangGraph • Groq • Tavily • RAG
            </p>
        </div>
        """)
        
        # ── MAIN LAYOUT ────────────────────────────
        with gr.Row():
            
            # ── LEFT: CHAT ─────────────────────────
            with gr.Column(scale=3):
                
                gr.Markdown("### 💬 Chat with Your AI Consultant")
                
                # Chat interface
                chatbot = gr.Chatbot(
                    value      = [],
                    height     = 500,
                    label      = "Conversation",
                    show_label = False,
                )
                
                # Input row
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder = (
                            "Ask anything... e.g. 'I want to study "
                            "AI in Germany' or 'Find universities "
                            "under $15,000'"
                        ),
                        label      = "",
                        scale      = 5,
                        show_label = False,
                        lines      = 1,
                    )
                    send_btn = gr.Button(
                        "Send ▶",
                        variant = "primary",
                        scale   = 1,
                    )
                
                # Quick action buttons
                gr.Markdown("**Quick Actions:**")
                with gr.Row():
                    profile_btn  = gr.Button(
                        "👤 My Profile", size="sm"
                    )
                    summary_btn  = gr.Button(
                        "📝 Summary", size="sm"
                    )
                    report_btn   = gr.Button(
                        "📊 Full Report", variant="primary",
                        size="sm"
                    )
                    clear_btn    = gr.Button(
                        "🗑️ Clear", variant="stop",
                        size="sm"
                    )
                
                # Example prompts
                gr.Markdown("**Try these examples:**")
                with gr.Row():
                    ex1 = gr.Button(
                        "🇩🇪 AI Masters in Germany",
                        size="sm"
                    )
                    ex2 = gr.Button(
                        "🇺🇸 Affordable US Programs",
                        size="sm"
                    )
                    ex3 = gr.Button(
                        "🌏 Find programs in Japan",
                        size="sm"
                    )
                
                with gr.Row():
                    ex4 = gr.Button(
                        "💰 Under $15,000 programs",
                        size="sm"
                    )
                    ex5 = gr.Button(
                        "🎓 Am I eligible?",
                        size="sm"
                    )
                    ex6 = gr.Button(
                        "📋 Recommend best options",
                        size="sm"
                    )
            
            # ── RIGHT: INFO PANELS ──────────────────
            with gr.Column(scale=2):
                
                with gr.Tabs():
                    
                    # Tab 1: Profile
                    with gr.Tab("👤 Profile"):
                        profile_display = gr.Markdown(
                            value = "Start a conversation to "
                                    "build your profile!"
                        )
                        refresh_profile_btn = gr.Button(
                            "🔄 Refresh Profile",
                            size="sm"
                        )
                    
                    # Tab 2: Report
                    with gr.Tab("📊 Report"):
                        report_display = gr.Markdown(
                            value = "Click 'Full Report' to "
                                    "generate recommendations!"
                        )
                    
                    # Tab 3: Currency
                    with gr.Tab("💱 PKR Rates"):
                        currency_display = gr.Markdown(
                            value = "Click refresh to see "
                                    "current rates!"
                        )
                        refresh_currency_btn = gr.Button(
                            "🔄 Get Live Rates",
                            size="sm"
                        )
                    
                    # Tab 4: Stats
                    with gr.Tab("📈 Stats"):
                        stats_display = gr.Markdown(
                            value = "Click refresh to see stats!"
                        )
                        refresh_stats_btn = gr.Button(
                            "🔄 Refresh Stats",
                            size="sm"
                        )
                    
                    # Tab 5: Live Log
                    with gr.Tab("📋 Live Log"):
                        gr.Markdown(
                            "**Real-time activity log** — "
                            "shows what's happening behind the scenes"
                        )
                        log_display = gr.Textbox(
                            value       = get_log_text(),
                            label       = "Activity Log",
                            lines       = 15,
                            max_lines   = 15,
                            interactive = False,
                            show_label  = False,
                        )
                        refresh_log_btn = gr.Button(
                            "🔄 Refresh Log",
                            size="sm"
                        )

        # ── SYSTEM INFO BAR ─────────────────────────
        gr.HTML(f"""
        <div style="margin-top:15px; padding:10px; 
                    background:#f5f5f5; border-radius:8px;
                    font-size:0.85em; color:#666;
                    display:flex; gap:20px; flex-wrap:wrap;">
            <span>🔑 Session: <code>{SESSION_ID}</code></span>
            <span>🤖 Model: LLaMA 3.3 70B (Groq)</span>
            <span>🔍 RAG: FAISS + HuggingFace</span>
            <span>🔄 Workflow: LangGraph</span>
            <span>🌐 Live Search: Tavily</span>
            <span>💱 Currency: Open Exchange Rates</span>
            <span>🔌 MCP: Filesystem + Database</span>
        </div>
        """)
        
        # ── EVENT HANDLERS ──────────────────────────
        
        def respond(message, history):
            """Handles chat message."""
            if not message.strip():
                return history, "", get_log_text()
            
            response = chat(message, history)
            
            history.append({
                "role": "user",
                "content": message
            })
            history.append({
                "role": "assistant", 
                "content": response
            })
            return history, "", get_log_text()
        
        def use_example(example_text):
            """Fills input with example text."""
            return example_text
        
        def clear_chat():
            result, _ = clear_all()
            return [], result
        
        # Send message on button click
        send_btn.click(
            fn      = respond,
            inputs  = [msg_input, chatbot],
            outputs = [chatbot, msg_input, log_display],
        )
        
        # Send message on Enter key
        msg_input.submit(
            fn      = respond,
            inputs  = [msg_input, chatbot],
            outputs = [chatbot, msg_input, log_display],
        )
        
        # Profile button
        profile_btn.click(
            fn      = lambda h: (h, get_profile()),
            inputs  = [chatbot],
            outputs = [chatbot, profile_display],
        )
        
        # Summary button
        summary_btn.click(
            fn      = lambda h: (h, get_summary()),
            inputs  = [chatbot],
            outputs = [chatbot, profile_display],
        )
        
        # Report button
        report_btn.click(
            fn      = generate_report,
            outputs = [report_display],
        )
        
        # Clear button
        clear_btn.click(
            fn      = clear_chat,
            outputs = [chatbot, profile_display],
        )
        
        # Refresh profile
        refresh_profile_btn.click(
            fn      = get_profile,
            outputs = [profile_display],
        )
        
        # Refresh currency
        refresh_currency_btn.click(
            fn      = get_currency_info,
            outputs = [currency_display],
        )
        
        # Refresh stats
        refresh_stats_btn.click(
            fn      = get_stats,
            outputs = [stats_display],
        )

        # Refresh log
        refresh_log_btn.click(
            fn      = get_log_text,
            outputs = [log_display],
        )
        
        # Example buttons
        ex1.click(fn=lambda: "Find AI Masters programs in Germany",
                  outputs=[msg_input])
        ex2.click(fn=lambda: "Show me affordable AI programs in USA under $15000",
                  outputs=[msg_input])
        ex3.click(fn=lambda: "Find AI Masters programs in Japan",
                  outputs=[msg_input])
        ex4.click(fn=lambda: "What universities can I afford with $15000 budget?",
                  outputs=[msg_input])
        ex5.click(fn=lambda: "Which universities am I eligible for?",
                  outputs=[msg_input])
        ex6.click(fn=lambda: "Recommend the best universities for me",
                  outputs=[msg_input])
    
    return demo

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        inbrowser   = True,
        theme       = gr.themes.Soft(
            primary_hue   = "blue",
            secondary_hue = "indigo",
        ),
    )