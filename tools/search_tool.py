# tools/search_tool.py
# Tavily API Integration
# Provides live web search for universities worldwide
# Used by Research Agent when CSV doesn't have enough info

import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

class TavilySearchTool:
    """
    Live web search tool powered by Tavily API.
    
    Tavily is built specifically for AI agents.
    Unlike Google, it returns clean structured results
    that are easy for AI to process.
    
    Used when:
    - Student asks about a country not in CSV
    - Student needs latest deadlines/fees
    - Student asks about specific university not in CSV
    - We need to verify/update information
    """
    
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        
        if not api_key:
            print("⚠️  Tavily API key not found!")
            self.client  = None
            self.enabled = False
        else:
            self.client  = TavilyClient(api_key=api_key)
            self.enabled = True
            print("✅ Tavily Search Tool initialized")
    
    # ─────────────────────────────────────────
    # CORE SEARCH METHOD
    # ─────────────────────────────────────────
    
    def search(self, query: str, max_results: int = 5) -> dict:
        """
        Performs a live web search.
        
        Args:
            query: search query string
            max_results: number of results to return
        
        Returns:
            dict with search results
        """
        
        if not self.enabled:
            return {
                "success": False,
                "message": "Tavily not configured",
                "results": [],
            }
        
        try:
            # Search the web
            response = self.client.search(
                query       = query,
                max_results = max_results,
                search_depth = "advanced",  # deeper search
            )
            
            # Extract and clean results
            results = []
            for r in response.get("results", []):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("url", ""),
                    "content": r.get("content", ""),
                })
            
            return {
                "success":     True,
                "query":       query,
                "results":     results,
                "count":       len(results),
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": str(e),
                "results": [],
            }
    
    # ─────────────────────────────────────────
    # UNIVERSITY SPECIFIC SEARCHES
    # ─────────────────────────────────────────
    
    def search_universities(self, field: str, country: str,
                             degree: str = "Masters",
                             language: str = "English") -> dict:
        """
        Searches for universities offering a specific program.
        
        Args:
            field:    field of study (e.g. "Artificial Intelligence")
            country:  target country
            degree:   degree type (default: Masters)
            language: language of instruction
        
        Returns:
            dict with university search results
        """
        
        query = (
            f"{language} taught {degree} in {field} "
            f"in {country} for international students 2025 2026"
        )
        
        print(f"🌐 Tavily searching: {query}")
        return self.search(query, max_results=5)
    
    def search_university_details(self, university: str,
                                   program: str) -> dict:
        """
        Gets detailed info about a specific university program.
        
        Args:
            university: university name
            program:    program name
        
        Returns:
            dict with detailed program information
        """
        
        query = (
            f"{university} {program} admission requirements "
            f"international students tuition fees deadline 2025 2026"
        )
        
        print(f"🌐 Tavily details: {university}")
        return self.search(query, max_results=3)
    
    def search_application_process(self, university: str,
                                    country: str) -> dict:
        """
        Searches for application process details.
        
        Args:
            university: university name
            country:    country of university
        
        Returns:
            dict with application process info
        """
        
        query = (
            f"{university} {country} application process "
            f"international students how to apply portal 2025"
        )
        
        print(f"🌐 Tavily application process: {university}")
        return self.search(query, max_results=3)
    
    def search_scholarships(self, country: str,
                             field: str) -> dict:
        """
        Searches for scholarship opportunities.
        
        Args:
            country: target country
            field:   field of study
        
        Returns:
            dict with scholarship information
        """
        
        query = (
            f"scholarships for international students "
            f"{field} {country} 2025 2026 funded"
        )
        
        print(f"🌐 Tavily scholarships: {country}")
        return self.search(query, max_results=5)
    
    def format_results_for_ai(self, search_results: dict) -> str:
        """
        Converts search results into clean text for AI.
        
        Args:
            search_results: results from search()
        
        Returns:
            formatted string for AI context
        """
        
        if not search_results.get("success"):
            return "Live search unavailable."
        
        if not search_results.get("results"):
            return "No live results found."
        
        formatted = "LIVE WEB SEARCH RESULTS:\n"
        formatted += "=" * 40 + "\n\n"
        
        for i, r in enumerate(search_results["results"], 1):
            formatted += f"Source {i}: {r['title']}\n"
            formatted += f"URL: {r['url']}\n"
            formatted += f"Content: {r['content'][:500]}...\n"
            formatted += "-" * 30 + "\n\n"
        
        return formatted