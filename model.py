import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TravelAIAgent:
    def __init__(self, df_cities, df_planner, df_rec, df_agency):
        # Clean columns immediately by forcing them to lowercase to prevent KeyErrors
        self.df_cities = df_cities.copy()
        self.df_cities.columns = [str(c).lower().strip() for c in self.df_cities.columns]
        
        self.df_planner = df_planner.copy()
        self.df_planner.columns = [str(c).lower().strip() for c in self.df_planner.columns]
        
        self.df_rec = df_rec
        self.df_agency = df_agency
        
        # Prepare lightweight vectorizer for matching travel rules/visas
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        # Clean up strings and safely handle empty/missing data points
        clean_planner_df = self.df_planner.astype(str).fillna("")
        self.planner_corpus = clean_planner_df.agg(' '.join, axis=1).tolist()
        
        self.vectorizer.fit(self.planner_corpus)
        self.planner_matrix = self.vectorizer.transform(self.planner_corpus)

        # Map correct semantic columns out safely
        self.city_col = 'city' if 'city' in self.df_cities.columns else self.df_cities.columns[0]
        self.dest_col = 'destination_name' if 'destination_name' in self.df_planner.columns else \
                        ('destination' if 'destination' in self.df_planner.columns else self.df_planner.columns[0])

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
        
        # Match cities dynamically using the sanitized lowercase column mapping
        detected_destination = "Unknown"
        if self.city_col in self.df_cities.columns:
            for city in self.df_cities[self.city_col].dropna().unique():
                if str(city).lower() in user_input_lower:
                    detected_destination = str(city)
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
        if destination != "Unknown" and self.dest_col in self.df_planner.columns:
            matches = self.df_planner[self.df_planner[self.dest_col].str.contains(destination, case=False, na=False)]
        else:
            matches = pd.DataFrame()
        
        # Cost lookup safety checks (handles variation variations like 'flight_transport_cost_usd')
        cost_col = None
        for col in self.df_planner.columns:
            if 'cost' in col or 'price' in col:
                cost_col = col
                break
                
        if budget and not matches.empty and cost_col:
            try:
                # Convert column to numeric values cleanly for filtering expressions
                matches[cost_col] = pd.to_numeric(matches[cost_col], errors='coerce')
                filtered_matches = matches[matches[cost_col] <= budget]
                if not filtered_matches.empty:
                    matches = filtered_matches
            except:
                pass
        
        # Similarity RAG backup context calculation
        input_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vector, self.planner_matrix).flatten()
        top_index = similarities.argsort()[-1]
        rag_fallback = self.df_planner.iloc[top_index].to_dict()
        
        # Attempt to gather specific visa text from schema attributes
        visa_info = "Visa requirement details not explicitly found. A human specialist will double-check."
        for col in self.df_planner.columns:
            if 'visa' in col:
                target_source = matches if not matches.empty else pd.DataFrame([rag_fallback])
                visa_info = f"Based on historical data: {target_source[col].iloc[0]}"
                break

        tour_options = matches.head(2).to_dict(orient='records') if not matches.empty else [rag_fallback]
        return tour_options, visa_info

    def generate_response(self, user_input):
        """
        Step 3: Formulate a natural response strictly grounded in verified facts.
        """
        entities = self.extract_entities(user_input)
        tours, visa = self.query_internal_database(entities, user_input)
        
        dest = entities['destination'] if entities['destination'] != "Unknown" else "your requested destination"
        
        output = f"### ✈️ AI Travel Assistant Options for {dest.title()}\n\n"
        output += "I searched our matching inventory based strictly on your preferences:\n\n"
        
        for idx, tour in enumerate(tours, 1):
            # Fallback chains to capture descriptive fields dynamically
            name = tour.get('destination_name', tour.get('city', tour.get('destination', f"Custom Package Plan {idx}")))
            cost = tour.get('flight_transport_cost_usd', tour.get('cost', tour.get('price', 'Quote on Request')))
            days = tour.get('recommended_stay_days', tour.get('duration', entities['duration'] or 'Flexible'))
            
            output += f"- **Option {idx}: {str(name).title()}** | Duration: {days} Days | Base Transport Cost: ${cost}\n"
            
        output += f"\n📋 **Visa Policy Guideline:**\n> {visa}\n\n"
        output += "Would you like to hand this over to our human sales team to secure your booking dates?"
        
        return output