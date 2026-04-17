# graph/workflow.py
# Part F: LangGraph State Machine Workflow
# This controls HOW agents talk to each other
# using a proper state graph with nodes and edges

# ─────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List, Optional
import json

# ─────────────────────────────────────────
# STATE DEFINITION
# ─────────────────────────────────────────

class UniversityFinderState(TypedDict):
    """
    This is the STATE of our workflow.
    
    Think of state like a shared whiteboard that all
    agents can read from and write to.
    
    Every node (agent) receives this state and
    returns an updated version of it.
    """
    
    # Conversation
    user_input:          str           # current message from student
    chat_history:        List          # full conversation history
    
    # Student info
    student_profile:     dict          # extracted profile data
    profile_complete:    bool          # is profile fully collected?
    
    # Agent outputs
    profile_response:    Optional[str] # what Profile Agent said
    research_results:    Optional[str] # what Research Agent found
    advisor_response:    Optional[str] # what Advisor Agent recommended
    
    # Workflow control
    current_node:        str           # which node is currently active
    next_node:           str           # which node to go to next
    final_response:      Optional[str] # final answer to show student
    
    # Research quality
    research_found:      bool          # did research find results?
    search_attempts:     int           # how many times we searched

# ─────────────────────────────────────────
# WORKFLOW CLASS
# ─────────────────────────────────────────

class UniversityFinderWorkflow:
    """
    The LangGraph workflow that orchestrates all 3 agents.
    
    Nodes = agents or processing steps
    Edges = connections between nodes
    Conditional edges = smart routing based on state
    """
    
    def __init__(self, profile_agent, research_agent, 
                 advisor_agent, memory_manager):
        """
        Args:
            profile_agent:  ProfileAgent instance
            research_agent: ResearchAgent instance  
            advisor_agent:  AdvisorAgent instance
            memory_manager: MemoryManager instance
        """
        
        self.profile_agent  = profile_agent
        self.research_agent = research_agent
        self.advisor_agent  = advisor_agent
        self.memory         = memory_manager
        
        # Build the graph
        self.graph = self._build_graph()
        
        print("✅ LangGraph workflow initialized")
    
    # ─────────────────────────────────────────
    # NODE DEFINITIONS
    # Each node is a function that:
    # - receives the current state
    # - does some work
    # - returns updated state
    # ─────────────────────────────────────────
    
    def profile_node(self, state: UniversityFinderState) -> dict:
        """
        NODE 1: Profile Collection
        
        Handles the student's message using Profile Agent.
        Collects and updates student information.
        """
        
        print("\n[🧑 Profile Node Active]")
        
        user_input      = state["user_input"]
        chat_history    = state["chat_history"]
        student_profile = state["student_profile"]
        
        # Call Profile Agent
        result = self.profile_agent.chat(
            user_input, chat_history, student_profile
        )
        
        response         = result["response"]
        profile_complete = result["profile_complete"]
        
        # If profile just completed, extract structured data
        if profile_complete:
            print("✅ Profile collection complete!")
            # Extract profile from conversation
            updated_profile = self.memory.extract_and_save_profile(
                chat_history + [
                    HumanMessage(content=user_input),
                    AIMessage(content=response)
                ]
            )
        else:
            updated_profile = student_profile
        
        # Determine next node
        if profile_complete:
            next_node = "research"  # move to research
        else:
            next_node = "end"       # keep collecting (show response)
        
        return {
            "profile_response": response,
            "student_profile":  updated_profile,
            "profile_complete": profile_complete,
            "final_response":   response,
            "current_node":     "profile",
            "next_node":        next_node,
        }
    
    def research_node(self, state: UniversityFinderState) -> dict:
        """
        NODE 2: University Research
        
        Searches for relevant universities using Research Agent.
        Uses RAG database to find matches.
        """
        
        print("\n[🔍 Research Node Active]")
        
        user_input      = state["user_input"]
        chat_history    = state["chat_history"]
        student_profile = state["student_profile"]
        search_attempts = state.get("search_attempts", 0)
        
        # Decide search strategy based on attempts
        if search_attempts == 0:
            # First attempt: search based on user input
            research_results = self.research_agent.research(
                user_input, chat_history, student_profile
            )
        else:
            # Retry: broader search using full profile
            print("🔄 Retrying with broader search...")
            research_results = self.research_agent.research_by_profile(
                student_profile, chat_history
            )
        
        # Check if research found anything useful
        research_found = len(research_results) > 100  # basic check
        
        # Determine next node
        if research_found:
            next_node = "advisor"  # found results, go to advisor
        elif search_attempts < 2:
            next_node = "research" # retry search
        else:
            next_node = "advisor"  # give up retrying, go to advisor anyway
        
        return {
            "research_results": research_results,
            "research_found":   research_found,
            "search_attempts":  search_attempts + 1,
            "current_node":     "research",
            "next_node":        next_node,
        }
    
    def advisor_node(self, state: UniversityFinderState) -> dict:
        """
        NODE 3: University Recommendations
        
        Takes research results and student profile to
        generate personalized recommendations.
        """
        
        print("\n[🎯 Advisor Node Active]")
        
        user_input       = state["user_input"]
        chat_history     = state["chat_history"]
        student_profile  = state["student_profile"]
        research_results = state.get("research_results", "")
        
        # Generate recommendations
        advisor_response = self.advisor_agent.advise(
            user_input,
            chat_history,
            student_profile,
            research_results
        )
        
        return {
            "advisor_response": advisor_response,
            "final_response":   advisor_response,
            "current_node":     "advisor",
            "next_node":        "end",
        }
    
    def router_node(self, state: UniversityFinderState) -> dict:
        """
        NODE 0: Smart Router
        
        Decides which agent should handle the message.
        This is the entry point of the graph.
        """
        
        print("\n[🔀 Router Node Active]")
        
        user_input       = state["user_input"]
        student_profile  = state.get("student_profile", {})
        profile_complete = state.get("profile_complete", False)
        
        user_lower = user_input.lower()
        
        # Keywords that mean student wants research
        research_keywords = [
            "find", "search", "show", "list", "what universities",
            "which universities", "university", "program", "programs",
            "masters", "eligible", "qualify", "apply", "fees"
        ]
        
        # Keywords that mean student wants advice
        advisor_keywords = [
            "recommend", "best", "top", "advice", "suggest",
            "should i", "compare", "rank", "shortlist", "report"
        ]
        
        # Routing logic
        if any(kw in user_lower for kw in advisor_keywords):
            next_node = "research"  # research first then advisor
            print("→ Routing to: Research → Advisor")
            
        elif any(kw in user_lower for kw in research_keywords):
            if profile_complete or student_profile:
                next_node = "research"
                print("→ Routing to: Research")
            else:
                next_node = "profile"
                print("→ Routing to: Profile (need info first)")
                
        elif not profile_complete:
            # Check completeness
            completeness = self.profile_agent.assess_profile_completeness(
                student_profile
            )
            if not completeness["is_complete"]:
                next_node = "profile"
                print(f"→ Routing to: Profile "
                      f"({completeness['completeness_percent']}% complete)")
            else:
                next_node = "research"
                print("→ Routing to: Research")
        else:
            next_node = "profile"
            print("→ Routing to: Profile (default)")
        
        return {
            "current_node": "router",
            "next_node":    next_node,
        }
    
    # ─────────────────────────────────────────
    # CONDITIONAL EDGE FUNCTIONS
    # These decide which node to go to NEXT
    # based on the current state
    # ─────────────────────────────────────────
    
    def route_from_router(self, state: UniversityFinderState) -> str:
        """After router, where do we go?"""
        return state.get("next_node", "profile")
    
    def route_from_profile(self, state: UniversityFinderState) -> str:
        """After profile node, where do we go?"""
        next_node = state.get("next_node", "end")
        if next_node == "research":
            return "research"
        return END
    
    def route_from_research(self, state: UniversityFinderState) -> str:
        """After research node, where do we go?"""
        next_node    = state.get("next_node", "advisor")
        search_attempts = state.get("search_attempts", 0)
        
        if next_node == "research" and search_attempts < 2:
            return "research"  # retry
        return "advisor"       # always go to advisor after research
    
    def route_from_advisor(self, state: UniversityFinderState) -> str:
        """After advisor, always end."""
        return END
    
    # ─────────────────────────────────────────
    # BUILD THE GRAPH
    # ─────────────────────────────────────────
    
    def _build_graph(self):
        """
        Builds the LangGraph state machine.
        
        This is where we:
        1. Create the graph with our state type
        2. Add nodes (agents)
        3. Add edges (connections)
        4. Set entry point
        5. Compile
        """
        
        # Create graph with our state type
        graph = StateGraph(UniversityFinderState)
        
        # ── ADD NODES ──────────────────────────────
        # Each node is a function that processes state
        graph.add_node("router",   self.router_node)
        graph.add_node("profile",  self.profile_node)
        graph.add_node("research", self.research_node)
        graph.add_node("advisor",  self.advisor_node)
        
        # ── SET ENTRY POINT ────────────────────────
        # Every conversation starts at the router
        graph.set_entry_point("router")
        
        # ── ADD CONDITIONAL EDGES ──────────────────
        # These are smart connections based on state
        
        # From router: go to profile, research, or advisor
        graph.add_conditional_edges(
            "router",
            self.route_from_router,
            {
                "profile":  "profile",
                "research": "research",
                "advisor":  "advisor",
            }
        )
        
        # From profile: either end (still collecting)
        # or go to research (profile complete)
        graph.add_conditional_edges(
            "profile",
            self.route_from_profile,
            {
                "research": "research",
                END:        END,
            }
        )
        
        # From research: retry or go to advisor
        graph.add_conditional_edges(
            "research",
            self.route_from_research,
            {
                "research": "research",
                "advisor":  "advisor",
            }
        )
        
        # From advisor: always end
        graph.add_conditional_edges(
            "advisor",
            self.route_from_advisor,
            {
                END: END,
            }
        )
        
        # ── COMPILE ────────────────────────────────
        return graph.compile()
    
    # ─────────────────────────────────────────
    # RUN THE WORKFLOW
    # ─────────────────────────────────────────
    
    def run(self, user_input: str) -> dict:
        """
        Runs the complete workflow for one user message.
        
        Args:
            user_input: message from student
        
        Returns:
            dict with final_response and updated state
        """
        
        # Build initial state
        initial_state = UniversityFinderState(
            user_input       = user_input,
            chat_history     = self.memory.get_chat_history(),
            student_profile  = self.memory.get_profile(),
            profile_complete = self._check_profile_complete(),
            profile_response = None,
            research_results = None,
            advisor_response = None,
            current_node     = "start",
            next_node        = "router",
            final_response   = None,
            research_found   = False,
            search_attempts  = 0,
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        return final_state
    
    def _check_profile_complete(self) -> bool:
        """Checks if current profile is complete."""
        profile = self.memory.get_profile()
        if not profile:
            return False
        completeness = self.profile_agent.assess_profile_completeness(
            profile
        )
        return completeness["is_complete"]
    
    def visualize(self):
        """
        Prints a text visualization of the graph.
        Shows all nodes and edges.
        """
        print("""
╔══════════════════════════════════════╗
║     LANGGRAPH WORKFLOW DIAGRAM       ║
╠══════════════════════════════════════╣
║                                      ║
║  [START]                             ║
║     ↓                                ║
║  [🔀 Router Node]                    ║
║     ↓                                ║
║  ┌──────────────────────┐            ║
║  │ Profile complete?    │            ║
║  │ Research keywords?   │            ║
║  │ Advisor keywords?    │            ║
║  └──────────────────────┘            ║
║     ↓           ↓           ↓        ║
║  [Profile]  [Research]  [Advisor]    ║
║     ↓           ↓           ↓        ║
║  Complete?  Found?      Always       ║
║     ↓ YES       ↓ YES   ends         ║
║  [Research] [Advisor]                ║
║     ↓           ↓                    ║
║  [Advisor]   [END]                   ║
║     ↓                                ║
║   [END]                              ║
║                                      ║
╚══════════════════════════════════════╝
        """)