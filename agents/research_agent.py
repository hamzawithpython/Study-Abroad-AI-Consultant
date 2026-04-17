# agents/research_agent.py
# Research Agent — now with Tavily live search!

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import json

load_dotenv()

class ResearchAgent:
    """
    Research Agent with both RAG and Live Search.
    
    Search Strategy:
    1. Always search local CSV first (fast)
    2. If results insufficient OR country not in CSV
       → use Tavily for live web search
    3. Combine both results for best answer
    """
    
    def __init__(self, rag_pipeline, search_tool=None):
        """
        Args:
            rag_pipeline: RAGPipeline instance
            search_tool:  TavilySearchTool instance (optional)
        """
        
        self.rag    = rag_pipeline
        self.tavily = search_tool  # NEW: Tavily search tool
        
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3
        )
        
        self.system_prompt = """
You are the Research Agent for an international university 
admissions system. Your ONLY job is to research universities.

YOUR RESPONSIBILITIES:
1. Search for relevant universities based on student profile
2. Find detailed program information
3. Identify admission requirements
4. Find application processes and deadlines
5. Note country-specific requirements

COUNTRY-SPECIFIC KNOWLEDGE:
- Germany: No tuition at public universities (~€315 semester fee)
            Applications through uni-assist.de for internationals
            Winter semester: October, Summer: April
            
- USA:      Direct university portal applications
            May require GRE (check each university)
            F-1 student visa required
            
- UK:       University portal or UCAS
            Graduate Route visa after graduation
            Usually 1-year Master's programs
            
- Canada:   Direct university applications
            Study permit required
            
- Netherlands: Studielink portal
               English-taught programs widely available

WHEN YOU HAVE LIVE SEARCH RESULTS:
- Prioritize live results for deadlines and current fees
- Use CSV data for baseline requirements
- Mention source when using live data

ALWAYS INCLUDE:
- Program name, tuition, requirements
- Application portal and deadline
- Scholarship opportunities
- Country specific requirements

LOCAL DATABASE RESULTS:
{rag_context}

LIVE WEB SEARCH RESULTS:
{live_search_context}

STUDENT PROFILE:
{student_profile}
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{user_input}"),
        ])
        
        self.chain = self.prompt | self.llm
        
        print("✅ Research Agent initialized")
    
    def research(self, user_input: str, chat_history: list,
                 student_profile: dict) -> str:
        """
        Researches universities using BOTH CSV and Tavily.
        """
        
        # Step 1: Search local CSV (RAG)
        search_query = self._build_search_query(
            user_input, student_profile
        )
        print(f"🔍 RAG searching: {search_query}")
        
        results     = self.rag.search_universities(
            search_query, k=6
        )
        rag_context = self.rag.get_context_for_ai(results)
        
        # Step 2: Live search with Tavily
        live_context = self._get_live_search(
            user_input, student_profile
        )
        
        profile_str = json.dumps(student_profile, indent=2) \
                      if student_profile else "No profile yet."
        
        # Step 3: AI combines both sources
        response = self.chain.invoke({
            "user_input":          user_input,
            "chat_history":        chat_history,
            "rag_context":         rag_context,
            "live_search_context": live_context,
            "student_profile":     profile_str,
        })
        
        return response.content
    
    def research_by_profile(self, student_profile: dict,
                             chat_history: list) -> str:
        """
        Full research based on student profile.
        Uses both RAG and Tavily.
        """
        
        # Build query from profile
        query_parts = []
        
        if student_profile.get("target_field"):
            query_parts.append(student_profile["target_field"])
        if student_profile.get("target_degree"):
            query_parts.append(student_profile["target_degree"])
        if student_profile.get("target_countries"):
            countries = student_profile["target_countries"]
            if isinstance(countries, list):
                query_parts.append(" ".join(countries))
            else:
                query_parts.append(str(countries))
        if student_profile.get("budget_tuition_usd"):
            query_parts.append(
                f"under ${student_profile['budget_tuition_usd']}"
            )
        
        search_query = " ".join(query_parts) if query_parts \
                       else "Masters Artificial Intelligence"
        
        print(f"🔍 Profile search: {search_query}")
        
        # RAG search
        results     = self.rag.search_universities(
            search_query, k=8
        )
        rag_context = self.rag.get_context_for_ai(results)
        
        # Live search
        live_context = self._get_live_search_by_profile(
            student_profile
        )
        
        profile_str = json.dumps(student_profile, indent=2)
        
        research_prompt = f"""
Based on student profile and search results, present 
comprehensive research on suitable universities.

For each university include:
- Program name and university
- Annual tuition (USD and PKR if available)
- Minimum GPA and English requirements
- Application portal and process
- Application deadline
- Eligibility based on student profile
- Scholarships available

Student Profile:
{profile_str}

Local Database Results:
{rag_context}

Live Web Results:
{live_context}

Present findings clearly and professionally.
"""
        
        response = self.llm.invoke(research_prompt)
        return response.content
    
    def _get_live_search(self, user_input: str,
                          profile: dict) -> str:
        """
        Gets live search results from Tavily.
        Returns formatted string for AI context.
        """
        
        if not self.tavily or not self.tavily.enabled:
            return "Live search not available."
        
        # Extract search parameters from profile
        field   = profile.get("target_field", "")
        countries = profile.get("target_countries", [])
        
        if isinstance(countries, list):
            country = countries[0] if countries else ""
        else:
            country = str(countries)
        
        # Search for universities
        if field and country:
            results = self.tavily.search_universities(
                field=field, country=country
            )
        else:
            results = self.tavily.search(
                user_input + " university Masters program"
            )
        
        return self.tavily.format_results_for_ai(results)
    
    def _get_live_search_by_profile(self, 
                                     profile: dict) -> str:
        """Gets live search based on complete profile."""
        
        if not self.tavily or not self.tavily.enabled:
            return "Live search not available."
        
        field     = profile.get("target_field", 
                                 "Artificial Intelligence")
        countries = profile.get("target_countries", [])
        
        if isinstance(countries, list):
            country = " ".join(countries)
        else:
            country = str(countries)
        
        # Search for universities
        results = self.tavily.search_universities(
            field=field, country=country
        )
        live_text = self.tavily.format_results_for_ai(results)
        
        # Also search for scholarships
        if country:
            scholarship_results = self.tavily.search_scholarships(
                country, field
            )
            scholarship_text = self.tavily.format_results_for_ai(
                scholarship_results
            )
            live_text += "\n\nSCHOLARSHIP SEARCH:\n" + scholarship_text
        
        return live_text
    
    def _build_search_query(self, user_input: str,
                              profile: dict) -> str:
        """Builds rich search query."""
        
        query_parts = [user_input]
        
        if profile.get("target_field") and \
           profile["target_field"].lower() not in \
           user_input.lower():
            query_parts.append(profile["target_field"])
        
        if profile.get("target_countries"):
            countries = profile["target_countries"]
            country_str = " ".join(countries) \
                          if isinstance(countries, list) \
                          else str(countries)
            if country_str.lower() not in user_input.lower():
                query_parts.append(country_str)
        
        return " ".join(query_parts)