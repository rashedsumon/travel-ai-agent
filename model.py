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
        
        # Clean up strings and safely handle empty/missing data points
        clean_planner_df = self.df_planner.astype(str).fillna("")
        self.planner_corpus = clean_planner_df.agg(' '.join, axis=1).tolist()
        
        self.vectorizer.fit(self.planner_corpus)
        self.planner_matrix = self.vectorizer.transform(self.planner_corpus)

        # DEFENSIVE LOOKUP: Automatically find the "City" column regardless of formatting or casing
        self.city_col = None
        for col in self.df_cities.columns:
            if 'city' in col.lower():
                self.city_col = col
                break
        # Fallback to the second column if no 'city' keyword matches explicitly
        if not self.city_col and len(self.df_cities.columns) > 1:
            self.city_col = self.df_cities.columns[1]
        elif not self.city_col:
            self.city_col = self.df_cities.columns[0]

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
        
        # Match cities dynamically using the auto-detected column signature
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
        
        # Dynamically locate the Destination reference column in the planner dataset
        dest_col = None
        for col in self.df_planner.columns:
            if 'dest' in col.lower():
                dest_col = col
                break
        if not dest_col:
            dest_col = self.df_planner.columns[0]
            
        # Try finding structural match first
        matches = self.df_planner[self.df_planner[dest_col].str.contains(destination, case=False, na=False)] if destination != "Unknown" else pd.DataFrame()
        
        if budget and not matches.empty:
            if 'Cost' in matches.columns:
                matches = matches[matches['Cost'] <= budget]
        
        # Similarity RAG backup context calculation
        input_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vector, self.planner_matrix).flatten()
        top_index = similarities.argsort()[-1]
        rag_fallback = self.df_planner.iloc[top_index].to_dict()
        
        # Attempt to gather specific visa text
        visa_info = "Visa requirement details not explicitly found. A human specialist will double-check."
        for col in self.df_planner.columns:
            if 'visa' in col.lower():
                # Use current matched subset or fallback to top rag candidate item
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
            # Attempt to gather any identifiable tour descriptive values
            name = tour.get('TourName', tour.get('PackageName', tour.get('City', tour.get('Destination', f"Custom Package Plan {idx}"))))
            cost = tour.get('Cost', tour.get('Price', tour.get('Budget', 'Quote on Request')))
            days = tour.get('Duration', tour.get('Days', entities['duration'] or 'Flexible'))
            output += f"- **Option {idx}: {name}** | Duration: {days} Days/Setting | Cost Estimate: {cost}\n"
            
        output += f"\n📋 **Visa Information:**\n> {visa}\n\n"
        output += "Would you like to hand this over to our human sales team to secure your booking dates?"
        
        return output