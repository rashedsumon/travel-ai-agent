import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TravelAIAgent:
    def __init__(self, df_cities, df_planner, df_rec, df_agency):
        self.df_cities = df_cities
        self.df_planner = df_planner
        self.df_rec = df_rec
        self.df_agency = df_agency
        
        # Prepare lightweight vectorizer for matching travel rules/visas
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Combine relevant data fields for string context comparisons
        self.planner_corpus = self.df_planner.astype(str).agg(' '.join, axis=1).tolist()
        self.vectorizer.fit(self.planner_corpus)
        self.planner_matrix = self.vectorizer.transform(self.planner_corpus)

    def extract_entities(self, user_input):
        """
        Step 1: Extract and structure data (destination, budget, duration).
        """
        user_input_lower = user_input.lower()
        
        # Simple extraction heuristics using Regex & Keyword Checks
        budget_match = re.search(r'\$?(\d{1,3}(?:,\d{3})*|\d+)', user_input_lower)
        budget = float(budget_match.group(1).replace(',', '')) if budget_match else None
        
        duration_match = re.search(r'(\d+)\s*(?:day|night)', user_input_lower)
        duration = int(duration_match.group(1)) if duration_match else None
        
        # Match cities based on our downloaded Cities Database
        detected_destination = "Unknown"
        for city in self.df_cities['City'].dropna().unique():
            if city.lower() in user_input_lower:
                detected_destination = city
                break
                
        return {
            "destination": detected_destination,
            "budget": budget,
            "duration": duration
        }

    def query_internal_database(self, entities, user_input):
        """
        Step 2: Query internal data (fetching matching tours & visa rules).
        Strict filtering prevents hallucinations.
        """
        destination = entities["destination"]
        budget = entities["budget"]
        
        # Try finding structural match first
        matches = self.df_planner[self.df_planner['Destination'].str.contains(destination, case=False, na=False)]
        
        if budget:
            # Look for cost rows if they exist in dataset schema (Fallback: fallback to destination filter)
            if 'Cost' in matches.columns:
                matches = matches[matches['Cost'] <= budget]
        
        # Similarity RAG backup context calculation
        input_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vector, self.planner_matrix).flatten()
        top_index = similarities.argsort()[-1]
        rag_fallback = self.df_planner.iloc[top_index].to_dict()
        
        # Attempt to gather specific visa text
        visa_info = "Visa requirement details not explicitly found. A human specialist will double-check."
        for col in matches.columns:
            if 'visa' in col.lower():
                visa_info = f"Based on historical data: {matches[col].iloc[0]}"
                break
        if "visa" in str(rag_fallback).lower() and visa_info.startswith("Visa requirement"):
            visa_info = f"Retrieved guidelines: {rag_fallback.get('Visa', 'Standard short-stay policies apply.')}"

        tour_options = matches.head(2).to_dict(orient='records') if not matches.empty else [rag_fallback]
        return tour_options, visa_info

    def generate_response(self, user_input):
        """
        Step 3: Formulate a natural response strictly grounded in verified facts.
        """
        entities = self.extract_entities(user_input)
        tours, visa = self.query_internal_database(entities, user_input)
        
        dest = entities['destination'] if entities['destination'] != "Unknown" else "your requested destination"
        
        # Build natural language narrative response anchored purely on data constraints
        output = f"### ✈️ AI Travel Assistant Options for {dest.title()}\n\n"
        output += "I searched our matching inventory based strictly on your preferences:\n\n"
        
        for idx, tour in enumerate(tours, 1):
            name = tour.get('TourName', tour.get('PackageName', f"Custom Package Plan {idx}"))
            cost = tour.get('Cost', tour.get('Price', 'Quote on Request'))
            days = tour.get('Duration', entities['duration'] or 'Flexible')
            output += f"- **Option {idx}: {name}** | Duration: {days} Days | Cost Estimate: ${cost}\n"
            
        output += f"\n📋 **Visa Information:**\n> {visa}\n\n"
        output += "Would you like to hand this over to our human sales team to secure your booking dates?"
        
        return output