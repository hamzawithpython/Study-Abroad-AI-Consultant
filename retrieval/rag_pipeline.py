# retrieval/rag_pipeline.py
# Part D: RAG Pipeline
# Loads university data, creates embeddings, enables smart search

# ─────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────

import os
import pandas as pd                                      # read CSV files
from langchain_community.vectorstores import FAISS       # vector database
from langchain_huggingface import HuggingFaceEmbeddings  # free embeddings
from langchain_core.documents import Document                    # document format
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# RAG PIPELINE CLASS
# ─────────────────────────────────────────

class RAGPipeline:
    """
    RAG = Retrieval Augmented Generation
    
    What this does:
    1. Loads university data from CSV
    2. Converts each university into a text document
    3. Creates embeddings (numerical representations of text)
    4. Stores in FAISS vector database
    5. When student asks a question, finds most relevant universities
    
    Think of embeddings like this:
    - "AI program in Texas" and "Artificial Intelligence Masters Texas"
      will have SIMILAR embeddings even though words are different
    - This means search works by MEANING not just keywords
    """
    
    def __init__(self):
        
        self.vectorstore = None   # will hold our FAISS database
        self.universities_df = None  # will hold raw CSV data
        
        # ── EMBEDDINGS MODEL ───────────────────────
        # HuggingFace model that converts text to numbers
        # all-MiniLM-L6-v2 is small, fast and free
        print("⏳ Loading embeddings model (first time takes 1-2 mins)...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # small but powerful model
            model_kwargs={"device": "cpu"},   # run on CPU (no GPU needed)
            encode_kwargs={"normalize_embeddings": True}
        )
        
        print("✅ Embeddings model loaded")
        
        # Path to our CSV and saved vector database
        self.csv_path = "retrieval/data/universities.csv"
        self.vectorstore_path = "data/vectorstore"
        
        # Load or create the vector database
        self._initialize_vectorstore()
    
    # ─────────────────────────────────────────
    # INITIALIZE VECTOR DATABASE
    # ─────────────────────────────────────────
    
    def _initialize_vectorstore(self):
        """
        Loads existing vector database if available.
        Otherwise creates a new one from the CSV.
        
        We save the database to disk so we don't have to
        recreate it every time the program starts.
        """
        
        if os.path.exists(self.vectorstore_path):
            # Load existing database from disk
            print("⏳ Loading existing vector database...")
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # needed for local files
            )
            print("✅ Vector database loaded from disk")
        else:
            # Create new database from CSV
            print("⏳ Creating vector database from university data...")
            self._create_vectorstore()
    
    def _create_vectorstore(self):
        """
        Reads universities CSV and creates FAISS vector database.
        Each university becomes one Document in the database.
        """
        
        # ── READ CSV ───────────────────────────────
        self.universities_df = pd.read_csv(self.csv_path)
        print(f"✅ Loaded {len(self.universities_df)} universities from CSV")
        
        # ── CONVERT TO DOCUMENTS ───────────────────
        # Each university becomes a text Document
        # The text describes everything about that university
        documents = []
        
        for _, row in self.universities_df.iterrows():
            
            # Create a rich text description of each university
            # The richer the text, the better the search results
            content = f"""
University: {row['university_name']}
Country: {row['country']}
City: {row['city']}
Program: {row['program_name']}
Degree Type: {row['degree_type']}
Field: {row['field']}
Language of Instruction: {row['language']}
Tuition per Year (USD): ${row['tuition_usd_per_year']}
Application Portal: {row['application_portal']}
English Requirement: {row['english_requirement']}
Minimum GPA Required: {row['min_gpa']}
Minimum IELTS: {row['min_ielts']}
Minimum TOEFL: {row['min_toefl']}
Available Intakes: {row['intake']}
Application Deadline: {row['deadline']}
Program Duration: {row['duration_years']} years
Scholarships Available: {row['scholarships_available']}
Notes: {row['special_notes']}
            """.strip()
            
            # Create Document with content and metadata
            # Metadata lets us filter later (e.g. by country)
            doc = Document(
                page_content=content,
                metadata={
                    "university": row['university_name'],
                    "country":    row['country'],
                    "city":       row['city'],
                    "field":      row['field'],
                    "tuition":    row['tuition_usd_per_year'],
                    "min_gpa":    row['min_gpa'],
                    "min_ielts":  row['min_ielts'],
                    "intake":     row['intake'],
                }
            )
            documents.append(doc)
        
        # ── CREATE FAISS VECTOR DATABASE ───────────
        # This converts all documents to embeddings
        # and stores them in the FAISS index
        self.vectorstore = FAISS.from_documents(
            documents,
            self.embeddings
        )
        
        # ── SAVE TO DISK ───────────────────────────
        # So we don't recreate it every time
        os.makedirs("data", exist_ok=True)
        self.vectorstore.save_local(self.vectorstore_path)
        
        print(f"✅ Vector database created and saved with {len(documents)} universities")
    
    # ─────────────────────────────────────────
    # SEARCH UNIVERSITIES
    # ─────────────────────────────────────────
    
    def search_universities(self, query: str, k: int = 5):
        """
        Searches the vector database for relevant universities.
        
        Args:
            query: what to search for (e.g. "AI Masters in Texas under $18000")
            k: how many results to return (default 5)
        
        Returns:
            List of relevant university documents
        """
        
        if not self.vectorstore:
            return []
        
        # Similarity search finds the k most relevant documents
        # It works by meaning, not just keywords
        results = self.vectorstore.similarity_search(query, k=k)
        return results
    
    def search_by_profile(self, profile: dict, k: int = 8):
        """
        Searches universities using the student's saved profile.
        Builds a smart search query from profile data.
        
        Args:
            profile: student profile dictionary from MemoryManager
            k: number of results
        """
        
        # Build a rich search query from the profile
        query_parts = []
        
        if profile.get("target_field"):
            query_parts.append(profile["target_field"])
        
        if profile.get("target_degree"):
            query_parts.append(profile["target_degree"])
        
        if profile.get("target_countries"):
            countries = profile["target_countries"]
            if isinstance(countries, list):
                query_parts.append(" ".join(countries))
            else:
                query_parts.append(str(countries))
        
        if profile.get("budget_tuition_usd"):
            query_parts.append(f"tuition under ${profile['budget_tuition_usd']}")
        
        if profile.get("target_intake"):
            query_parts.append(profile["target_intake"])
        
        # Combine into one search query
        query = " ".join(query_parts)
        
        if not query.strip():
            query = "Masters in Artificial Intelligence"  # fallback
        
        print(f"🔍 Searching with query: {query}")
        return self.search_universities(query, k=k)
    
    def get_context_for_ai(self, results):
        """
        Converts search results into a formatted string
        that can be passed to the AI as context.
        
        Args:
            results: list of Documents from search
        
        Returns:
            formatted string with all university info
        """
        
        if not results:
            return "No relevant universities found in local database."
        
        context = "RELEVANT UNIVERSITIES FROM DATABASE:\n"
        context += "=" * 50 + "\n\n"
        
        for i, doc in enumerate(results, 1):
            context += f"UNIVERSITY {i}:\n"
            context += doc.page_content
            context += "\n\n" + "-" * 40 + "\n\n"
        
        return context
    
    def force_rebuild(self):
        """
        Forces a rebuild of the vector database.
        Use this if you update the CSV file.
        """
        import shutil
        if os.path.exists(self.vectorstore_path):
            shutil.rmtree(self.vectorstore_path)
            print("✅ Old vector database deleted")
        self._create_vectorstore()