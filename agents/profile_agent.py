# agents/profile_agent.py
# Profile Agent — collects and manages student information
# This agent's ONLY job is to have a conversation with the
# student and build a complete profile

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

class ProfileAgent:
    """
    The Profile Agent acts like a friendly admissions advisor
    whose ONLY job is to collect student information.
    
    It knows:
    - What questions to ask
    - When enough info is collected
    - How to handle vague or incomplete answers
    - How to confirm the profile with the student
    """
    
    def __init__(self):
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.7   # friendly conversational tone
        )
        
        # This agent's specific system prompt
        # Very focused on profile collection
        self.system_prompt = """
You are the Profile Collection Agent for an international university 
admissions system. Your ONLY job is to collect complete student 
profile information through friendly conversation.

YOUR RESPONSIBILITIES:
1. Collect all required profile information
2. Ask ONE question at a time — never multiple questions at once
3. Ask follow-up questions if answers are vague
4. Be friendly and encouraging
5. Confirm the profile when complete

INFORMATION YOU MUST COLLECT:
- Bachelor's degree subject
- University name and country
- GPA (and the scale e.g. 3.5/4.0 or 85%)
- Graduation year (completed or expected)
- Target field for Master's
- Preferred countries for study
- Annual budget (tuition fees)
- Initial budget (for visa, travel, setup costs)
- English proficiency (IELTS/TOEFL score OR exemption reason)
- Target intake (Fall/Winter/Spring and year)
- Career goals after graduation
- Any specific preferences (city, research vs coursework, etc.)

CONVERSATION RULES:
- If student gives vague answer like "good GPA" → ask for exact number
- If student says "Europe" → ask which specific countries
- If student mentions exemption from English test → ask for proof/reason
- If student skips something → come back to it naturally
- Never make assumptions about missing information

WHEN PROFILE IS COMPLETE:
- Summarize the entire profile clearly
- Ask student to confirm it's correct
- End your message with exactly: [PROFILE_COMPLETE]

CURRENT STUDENT PROFILE:
{current_profile}

Remember: You are collecting info only. Do NOT search for universities
or make recommendations — that's another agent's job.
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{user_input}"),
        ])
        
        self.chain = self.prompt | self.llm
        
        print("✅ Profile Agent initialized")
    
    def chat(self, user_input: str, chat_history: list, 
             current_profile: dict) -> dict:
        """
        Process a message from the student.
        
        Args:
            user_input: what the student said
            chat_history: full conversation history
            current_profile: profile collected so far
        
        Returns:
            dict with:
            - response: the agent's reply
            - profile_complete: True if profile is done
        """
        
        # Convert profile dict to readable string for the prompt
        profile_str = json.dumps(current_profile, indent=2) if current_profile \
                      else "No profile information collected yet."
        
        response = self.chain.invoke({
            "user_input":      user_input,
            "chat_history":    chat_history,
            "current_profile": profile_str,
        })
        
        reply = response.content
        
        # Check if profile is complete
        # Agent signals completion with [PROFILE_COMPLETE]
        profile_complete = "[PROFILE_COMPLETE]" in reply
        
        # Clean the signal from the response shown to student
        clean_reply = reply.replace("[PROFILE_COMPLETE]", "").strip()
        
        return {
            "response":        clean_reply,
            "profile_complete": profile_complete,
        }
    
    def assess_profile_completeness(self, profile: dict) -> dict:
        """
        Checks which required fields are missing from the profile.
        
        Returns:
            dict with:
            - is_complete: True if all required fields present
            - missing_fields: list of what's still needed
            - completeness_percent: how complete the profile is
        """
        
        required_fields = [
            "bachelor_degree",
            "university", 
            "gpa",
            "target_field",
            "target_countries",
            "budget_tuition_usd",
            "target_intake",
        ]
        
        important_fields = [
            "english_test",
            "english_exemption",
            "career_goal",
            "budget_initial_pkr",
        ]
        
        missing_required  = []
        missing_important = []
        
        for field in required_fields:
            if not profile.get(field):
                missing_required.append(field)
        
        for field in important_fields:
            if not profile.get(field):
                missing_important.append(field)
        
        total_fields    = len(required_fields) + len(important_fields)
        filled_fields   = total_fields - len(missing_required) - len(missing_important)
        completeness    = int((filled_fields / total_fields) * 100)
        
        return {
            "is_complete":          len(missing_required) == 0,
            "missing_required":     missing_required,
            "missing_important":    missing_important,
            "completeness_percent": completeness,
        }