# memory/memory_manager.py
# Part C: Memory System (Modern LangChain approach)

# ─────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import json
import os

load_dotenv()

# ─────────────────────────────────────────
# MEMORY MANAGER CLASS
# ─────────────────────────────────────────

class MemoryManager:
    """
    Manages all memory for the chatbot.
    
    - chat_history    = full conversation (buffer memory)
    - summary         = AI generated summary (summary memory)  
    - student_profile = structured JSON profile (persistent memory)
    """
    
    def __init__(self, fresh_start: bool = False):
        """
        Args:
            fresh_start: if True, ignores saved profile
                        Used for new sessions in Gradio
        """
        
        # Create data folder immediately
        os.makedirs("data", exist_ok=True)
        
        self.fresh_start = fresh_start
        
        # Set up AI model for generating summaries
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        
        # ── BUFFER MEMORY ──────────────────────────
        # Simple list of HumanMessage and AIMessage objects
        # This IS the modern LangChain way of doing buffer memory
        self.chat_history = []
        
        # ── SUMMARY MEMORY ─────────────────────────
        # Plain text summary, updated every 4 messages
        self.conversation_summary = "No conversation yet."
        
        # ── STUDENT PROFILE ────────────────────────
        # Structured dictionary of student info
        self.student_profile = {}
        
        # Load existing profile only if not fresh start
        if not fresh_start:
            self._load_profile_from_file()
            self._load_summary_from_file()
        else:
            self.student_profile = {}
            self.conversation_summary = "No conversation yet."
            print("🆕 Fresh session started")
        
        print("✅ Memory system initialized")
    
    # ─────────────────────────────────────────
    # SAVE A MESSAGE EXCHANGE
    # ─────────────────────────────────────────
    
    def save_message(self, human_message: str, ai_message: str):
        """
        Saves each exchange to chat history.
        Called after every message.
        """
        self.chat_history.append(HumanMessage(content=human_message))
        self.chat_history.append(AIMessage(content=ai_message))
    
    # ─────────────────────────────────────────
    # GET MEMORY
    # ─────────────────────────────────────────
    
    def get_chat_history(self):
        """Returns full conversation history as message objects."""
        return self.chat_history
    
    def get_summary(self):
        """Returns current conversation summary as text."""
        return self.conversation_summary
    
    # ─────────────────────────────────────────
    # UPDATE SUMMARY
    # ─────────────────────────────────────────
    
    def update_summary(self):
        """
        Uses AI to generate/update a summary of the conversation.
        Called every 4 messages.
        """
        if not self.chat_history:
            return
        
        # Build conversation text
        history_text = ""
        for msg in self.chat_history:
            if isinstance(msg, HumanMessage):
                history_text += f"Student: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Consultant: {msg.content}\n"
        
        # Ask AI to summarize
        summary_prompt = f"""
        Summarize this conversation between a student and a university 
        admissions consultant. Focus on:
        - What the student is looking for
        - Key profile details mentioned
        - What has been decided or confirmed
        
        Keep it concise but complete.
        
        Conversation:
        {history_text}
        
        Summary:
        """
        
        try:
            response = self.llm.invoke(summary_prompt)
            self.conversation_summary = response.content
            self._save_summary_to_file()
            print("✅ Summary updated\n")
        except Exception as e:
            print(f"⚠️ Summary update note: {e}")
    
    # ─────────────────────────────────────────
    # STUDENT PROFILE EXTRACTION
    # ─────────────────────────────────────────
    
    def extract_and_save_profile(self, conversation_history):
        """
        Uses AI to extract structured profile from conversation.
        Saves to JSON file for persistence.
        """
        
        # Build conversation text
        history_text = ""
        for msg in conversation_history:
            if isinstance(msg, HumanMessage):
                history_text += f"Student: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_text += f"Consultant: {msg.content}\n"
        
        extraction_prompt = f"""
        Based on this conversation, extract the student's profile.
        Return ONLY a valid JSON object with these fields 
        (use null if not mentioned):
        
        {{
            "name": null,
            "bachelor_degree": null,
            "university": null,
            "gpa": null,
            "gpa_scale": null,
            "graduation_year": null,
            "target_degree": null,
            "target_field": null,
            "target_countries": null,
            "budget_tuition_usd": null,
            "budget_initial_pkr": null,
            "english_test": null,
            "english_score": null,
            "english_exemption": null,
            "target_intake": null,
            "career_goal": null,
            "special_notes": null
        }}
        
        Conversation:
        {history_text}
        
        Return ONLY the JSON. No explanation. No markdown. No extra text.
        """
        
        try:
            response = self.llm.invoke(extraction_prompt)
            
            # Clean response
            json_text = response.content.strip()
            json_text = json_text.replace("```json", "").replace("```", "").strip()
            
            # Parse JSON
            extracted = json.loads(json_text)
            
            # Update profile with non-null values only
            for key, value in extracted.items():
                if value is not None:
                    self.student_profile[key] = value
            
            # Save to file
            self._save_profile_to_file()
            print("✅ Profile updated\n")
            
            return self.student_profile
            
        except Exception as e:
            import traceback
            print(f"⚠️ Profile extraction error: {e}")
            traceback.print_exc()
            return self.student_profile
    
    # ─────────────────────────────────────────
    # PROFILE DISPLAY
    # ─────────────────────────────────────────
    
    def get_profile(self):
        """Returns the raw profile dictionary."""
        return self.student_profile
    
    def get_profile_summary(self):
        """Returns a nicely formatted profile string."""
        if not self.student_profile:
            return "\n⚠️ No profile information collected yet. Have a conversation first!\n"
        
        summary = "\n📋 YOUR CURRENT PROFILE:\n"
        summary += "─" * 40 + "\n"
        
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
            "english_test":       "🗣️ English Test",
            "english_score":      "🗣️ English Score",
            "english_exemption":  "🗣️ English Exemption",
            "target_intake":      "📅 Target Intake",
            "career_goal":        "🚀 Career Goal",
            "special_notes":      "📝 Notes",
        }
        
        for field, label in field_labels.items():
            value = self.student_profile.get(field)
            if value:
                summary += f"{label}: {value}\n"
        
        summary += "─" * 40
        return summary
    
    # ─────────────────────────────────────────
    # FILE OPERATIONS
    # ─────────────────────────────────────────
    
    def _save_profile_to_file(self):
        """Saves profile to JSON file."""
        os.makedirs("data", exist_ok=True)
        with open("data/student_profile.json", "w") as f:
            json.dump(self.student_profile, f, indent=2)
    
    def _load_profile_from_file(self):
        """Loads profile from JSON file if it exists."""
        try:
            with open("data/student_profile.json", "r") as f:
                self.student_profile = json.load(f)
            if self.student_profile:
                print("✅ Previous student profile loaded from file")
        except FileNotFoundError:
            self.student_profile = {}
    
    def _save_summary_to_file(self):
        """Saves summary to text file."""
        os.makedirs("data", exist_ok=True)
        with open("data/conversation_summary.txt", "w") as f:
            f.write(self.conversation_summary)
    
    def _load_summary_from_file(self):
        """Loads summary from file if it exists."""
        try:
            with open("data/conversation_summary.txt", "r") as f:
                self.conversation_summary = f.read()
            print("✅ Previous summary loaded from file")
        except FileNotFoundError:
            self.conversation_summary = "No conversation yet."
    
    def clear_profile(self):
        """Clears student profile."""
        self.student_profile = {}
        if os.path.exists("data/student_profile.json"):
            os.remove("data/student_profile.json")
        print("✅ Profile cleared")
    
    def clear_all_memory(self):
        """Full reset — clears everything."""
        self.chat_history = []
        self.conversation_summary = "No conversation yet."
        self.clear_profile()
        if os.path.exists("data/conversation_summary.txt"):
            os.remove("data/conversation_summary.txt")
        print("✅ All memory cleared")