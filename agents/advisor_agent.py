# agents/advisor_agent.py
# Advisor Agent — matches student profile to universities
# and gives final personalized recommendations

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json

load_dotenv()

class AdvisorAgent:
    """
    The Advisor Agent is the final expert in the pipeline.
    
    It takes:
    - Student profile (from Profile Agent)
    - Research results (from Research Agent)
    
    And produces:
    - Eligibility assessment
    - Ranked recommendations
    - Application strategy
    - Personalized advice
    """
    
    def __init__(self):
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.5
        )
        
        self.system_prompt = """
You are the Senior Admissions Advisor for an international university 
admissions system. You give FINAL recommendations to students.

YOUR RESPONSIBILITIES:
1. Match student profile against university requirements
2. Determine eligibility for each university
3. Rank universities by fit (reach, match, safety)
4. Give honest, detailed advice
5. Suggest application strategy

ELIGIBILITY ASSESSMENT RULES:
- Compare student GPA against minimum GPA required
- Compare IELTS/TOEFL score against requirements
  (If student has exemption, check if university accepts it)
- Check if budget covers tuition fees
- Check if desired intake is available

UNIVERSITY CATEGORIES:
- REACH:  Student meets minimum but it's competitive
- MATCH:  Student comfortably meets requirements  
- SAFETY: Student clearly exceeds requirements

OUTPUT FORMAT:
For each recommended university provide:

🏛️ [University Name] — [REACH/MATCH/SAFETY]
📋 Program: [program name]
💰 Tuition: $[amount]/year
✅ Eligibility: [why they qualify or don't]
📅 Deadline: [deadline]
📝 How to Apply: [portal/process]
📎 Documents Needed: [list key documents]
⭐ Why This Fits You: [personalized reason]

After listing universities:
- Give TOP 3 recommendations with reasons
- Suggest application timeline
- Give honest advice about chances

STUDENT PROFILE:
{student_profile}

RESEARCH RESULTS:
{research_results}
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{user_input}"),
        ])
        
        self.chain = self.prompt | self.llm
        
        print("✅ Advisor Agent initialized")
    
    def advise(self, user_input: str, chat_history: list,
               student_profile: dict, research_results: str) -> str:
        """
        Generates personalized university recommendations.
        
        Args:
            user_input: student's question
            chat_history: conversation history
            student_profile: complete student profile
            research_results: findings from Research Agent
        
        Returns:
            personalized recommendations as string
        """
        
        profile_str = json.dumps(student_profile, indent=2) if student_profile \
                      else "No profile available."
        
        response = self.chain.invoke({
            "user_input":       user_input,
            "chat_history":     chat_history,
            "student_profile":  profile_str,
            "research_results": research_results,
        })
        
        return response.content
    
    def generate_full_report(self, student_profile: dict,
                              research_results: str,
                              chat_history: list) -> str:
        """
        Generates a complete advisory report for the student.
        Called when student asks for full recommendations.
        """
        
        profile_str = json.dumps(student_profile, indent=2)
        
        report_prompt = f"""
Generate a comprehensive university recommendation report for this student.

STUDENT PROFILE:
{profile_str}

RESEARCH FINDINGS:
{research_results}

Your report must include:

1. PROFILE SUMMARY
   - Quick summary of student's background and goals

2. ELIGIBLE UNIVERSITIES
   - List all universities student qualifies for
   - Categorize as REACH / MATCH / SAFETY
   - For each: fees, requirements, deadline, portal

3. NOT ELIGIBLE (and why)
   - List universities they don't qualify for
   - Explain what's missing (GPA too low, budget etc.)

4. TOP 3 RECOMMENDATIONS
   - Your best picks with detailed reasoning
   - Why each one suits this student specifically

5. APPLICATION STRATEGY
   - Suggested timeline
   - Documents to prepare
   - Tips for strong application

6. FINANCIAL SUMMARY
   - Cost breakdown for top choices
   - Scholarship opportunities

Be honest, specific, and genuinely helpful.
"""
        
        response = self.llm.invoke(report_prompt)
        return response.content
    
    def check_single_eligibility(self, student_profile: dict,
                                  university_info: str) -> str:
        """
        Checks if student is eligible for a specific university.
        Used when student asks about one particular university.
        """
        
        profile_str = json.dumps(student_profile, indent=2)
        
        check_prompt = f"""
Determine if this student is eligible for this university program.

Student Profile:
{profile_str}

University Information:
{university_info}

Provide:
1. ELIGIBLE or NOT ELIGIBLE
2. Reason for each requirement (GPA, English, Budget)
3. If not eligible, what they need to improve
4. If eligible, their chances (strong/moderate/borderline)
"""
        
        response = self.llm.invoke(check_prompt)
        return response.content