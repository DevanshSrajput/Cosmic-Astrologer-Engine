import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import random
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class CosmicAstrologerEngine:
    """Complete astrologer recommendation and AI reading system"""
    
    def __init__(self):
        """Initialize the cosmic engine with optimized setup"""
        print("🔮 Initializing Cosmic Astrologer Engine...")
        
        # Load astrologer data
        self.astrologers_data = self._create_astrologer_dataset()
        
        # Setup TF-IDF for fast, lightweight recommendations
        self._setup_recommendation_engine()
        
        # Initialize AI astrologer components
        self._setup_ai_astrologer()
        
        print("✨ Cosmic Engine Ready! Choose your path:")
        print("  1. Find Perfect Astrologer")
        print("  2. Get AI Astrologer Reading")
        print("  3. Run Demo & Analytics")
    
    def _create_astrologer_dataset(self) -> pd.DataFrame:
        """Create optimized astrologer dataset"""
        astrologers = [
            {
                "name": "Luna Starweaver",
                "specialties": ["love", "relationships", "soulmate", "marriage", "romance", "dating"],
                "description": "Expert in matters of the heart, twin flames, and romantic destiny. Specializes in love compatibility and relationship guidance through celestial insights.",
                "experience": 12, "rating": 4.9, "focus_area": "love"
            },
            {
                "name": "Sage Moonlight", 
                "specialties": ["career", "business", "success", "money", "professional", "work"],
                "description": "Career astrologer focusing on professional success, business ventures, and financial prosperity through planetary alignments.",
                "experience": 8, "rating": 4.7, "focus_area": "career"
            },
            {
                "name": "Crystal Visionheart",
                "specialties": ["health", "wellness", "healing", "mental", "vitality", "medical"],
                "description": "Holistic astrologer specializing in health predictions, wellness guidance, and spiritual healing through astrological remedies.",
                "experience": 15, "rating": 4.8, "focus_area": "health"
            },
            {
                "name": "Mystic Stardust",
                "specialties": ["family", "marriage", "children", "home", "domestic", "parents"],
                "description": "Family-focused astrologer helping with domestic issues, marriage compatibility, childbirth timing, and family harmony.",
                "experience": 20, "rating": 4.9, "focus_area": "family"
            },
            {
                "name": "Oracle Nightsky", 
                "specialties": ["spiritual", "meditation", "enlightenment", "karma", "past life", "soul"],
                "description": "Spiritual guide specializing in karmic patterns, past life readings, meditation guidance, and spiritual awakening.",
                "experience": 18, "rating": 4.6, "focus_area": "spiritual"
            },
            {
                "name": "Phoenix Cosmicwind",
                "specialties": ["travel", "relocation", "foreign", "adventure", "changes", "moving"],
                "description": "Expert in travel astrology, relocation guidance, foreign opportunities, and major life transitions.",
                "experience": 10, "rating": 4.5, "focus_area": "travel"
            },
            {
                "name": "Stellar Fortuneteller",
                "specialties": ["general", "predictions", "future", "guidance", "purpose", "destiny"],
                "description": "Generalist astrologer providing comprehensive life readings, future predictions, and overall life purpose guidance.",
                "experience": 25, "rating": 4.8, "focus_area": "general"
            }
        ]
        return pd.DataFrame(astrologers)
    
    def _setup_recommendation_engine(self):
        """Setup optimized TF-IDF recommendation engine"""
        # Create searchable text for each astrologer
        self.astrologer_texts = []
        for _, astrologer in self.astrologers_data.iterrows():
            text = " ".join(astrologer['specialties']) + " " + astrologer['description']
            self.astrologer_texts.append(text.lower())
        
        # Initialize TF-IDF with optimized parameters
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000,
            lowercase=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.astrologer_texts)
        print("🚀 Recommendation engine optimized with TF-IDF")
    
    def _setup_ai_astrologer(self):
        """Setup AI astrologer components"""
        self.zodiac_traits = {
            'aries': ['leadership', 'courage', 'energy'], 'taurus': ['stability', 'determination'], 
            'gemini': ['communication', 'adaptability'], 'cancer': ['nurturing', 'intuition'],
            'leo': ['confidence', 'creativity'], 'virgo': ['analysis', 'perfection'],
            'libra': ['harmony', 'balance'], 'scorpio': ['intensity', 'transformation'],
            'sagittarius': ['adventure', 'wisdom'], 'capricorn': ['ambition', 'discipline'],
            'aquarius': ['innovation', 'independence'], 'pisces': ['intuition', 'compassion']
        }
        
        self.house_meanings = {
            '1st': 'identity', '2nd': 'resources', '3rd': 'communication', '4th': 'home',
            '5th': 'creativity', '6th': 'health', '7th': 'relationships', '8th': 'transformation',
            '9th': 'wisdom', '10th': 'career', '11th': 'friendships', '12th': 'spirituality'
        }
        
        self.planets = {
            'sun': 'core identity', 'moon': 'emotions', 'mercury': 'communication',
            'venus': 'love', 'mars': 'action', 'jupiter': 'growth', 'saturn': 'discipline'
        }
        
        self.mystical_openings = [
            "The cosmic energies reveal", "Your celestial blueprint shows", "The stars whisper",
            "Ancient wisdom flows through", "Divine timing brings", "The universe aligns"
        ]
        
        self.guidance_phrases = [
            "Trust your intuition during this powerful time",
            "Embrace the transformative energies surrounding you", 
            "Your spiritual journey accelerates now",
            "Pay attention to synchronicities as divine messages",
            "This cosmic window opens new possibilities"
        ]
    
    def _calculate_percentage_relevance(self, score: float) -> int:
        """Convert cosine similarity to percentage relevance"""
        # Normalize cosine similarity (0-1) to percentage (0-100)
        # Apply a boost for better user experience
        if score <= 0:
            return 0
        
        # Enhanced scoring algorithm for better percentage distribution
        percentage = min(100, int((score ** 0.7) * 100))
        
        # Ensure minimum 5% for any positive match
        if percentage > 0 and percentage < 5:
            percentage = 5
            
        return percentage
    
    def find_astrologer(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find best matching astrologers using optimized search"""
        if not query.strip():
            return []
        
        # Transform query and calculate similarities
        query_vector = self.vectorizer.transform([query.lower()])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include relevant matches
                astrologer = self.astrologers_data.iloc[idx]
                raw_score = similarities[idx]
                percentage_score = self._calculate_percentage_relevance(raw_score)
                
                recommendations.append({
                    'name': astrologer['name'],
                    'specialties': astrologer['specialties'][:3],  # Show top 3
                    'description': astrologer['description'][:120] + "...",
                    'experience': astrologer['experience'],
                    'rating': astrologer['rating'],
                    'focus': astrologer['focus_area'],
                    'score_raw': round(raw_score, 3),
                    'score_percentage': percentage_score
                })
        
        return recommendations
    
    def get_ai_reading(self, input_text: str) -> str:
        """Generate AI astrologer reading"""
        if not input_text.strip():
            return "Please share your cosmic question for guidance."
        
        input_lower = input_text.lower()
        reading_parts = []
        
        # Mystical opening
        opening = random.choice(self.mystical_openings)
        reading_parts.append(f"{opening} powerful insights for your journey.")
        
        # Parse astrological elements
        found_elements = False
        
        # Check for zodiac signs
        for sign, traits in self.zodiac_traits.items():
            if sign in input_lower:
                trait = random.choice(traits)
                reading_parts.append(f"Your {sign.title()} energy emphasizes {trait}, guiding your path forward.")
                found_elements = True
        
        # Check for planets
        for planet, meaning in self.planets.items():
            if planet in input_lower:
                reading_parts.append(f"{planet.title()} influences your {meaning}, bringing significant developments.")
                found_elements = True
        
        # Check for houses
        house_pattern = r'(\d+)(?:st|nd|rd|th)?\s*house'
        houses = re.findall(house_pattern, input_lower)
        for house_num in houses:
            if house_num in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
                house_key = f"{house_num}{'st' if house_num == '1' else 'nd' if house_num == '2' else 'rd' if house_num == '3' else 'th'}"
                if house_key in self.house_meanings:
                    meaning = self.house_meanings[house_key]
                    reading_parts.append(f"The {house_key} house of {meaning} receives cosmic attention, promising growth.")
                    found_elements = True
        
        # Add guidance
        guidance = random.choice(self.guidance_phrases)
        reading_parts.append(guidance + ".")
        
        # Generic reading if no astrological elements found
        if not found_elements:
            generic_insights = [
                "The cosmic currents are shifting to support your highest good",
                "Your soul is ready for a powerful transformation and awakening",
                "Divine opportunities are manifesting in your life path",
                "The universe is conspiring to bring you exactly what you need"
            ]
            reading_parts.insert(1, random.choice(generic_insights) + ".")
        
        # Cosmic blessing
        blessings = [
            "May the stars illuminate your path ahead",
            "Trust in the cosmic plan unfolding for you", 
            "The universe supports your journey toward fulfillment"
        ]
        reading_parts.append(random.choice(blessings) + ".")
        
        return " ".join(reading_parts)
    
    def _get_relevance_emoji(self, percentage: int) -> str:
        """Get emoji based on relevance percentage"""
        if percentage >= 80:
            return "🔥"
        elif percentage >= 60:
            return "✨"
        elif percentage >= 40:
            return "⭐"
        elif percentage >= 20:
            return "🌟"
        else:
            return "💫"
    
    def print_recommendations(self, recommendations: List[Dict], query: str = ""):
        """Pretty print astrologer recommendations with percentage scores"""
        if not recommendations:
            print("❌ No relevant astrologers found. Try different keywords.")
            return
        
        print(f"\n🔮 {'Query: ' + query if query else 'Astrologer Recommendations'}")
        print("=" * 70)
        
        for i, rec in enumerate(recommendations, 1):
            relevance_emoji = self._get_relevance_emoji(rec['score_percentage'])
            print(f"\n{i}. ⭐ {rec['name']} ({rec['rating']}/5.0)")
            print(f"   🎯 Focus: {rec['focus'].title()} | 📅 {rec['experience']} years experience")
            print(f"   🔑 Specialties: {', '.join(rec['specialties'])}")
            print(f"   📝 {rec['description']}")
            print(f"   {relevance_emoji} Relevance: {rec['score_percentage']}% (score: {rec['score_raw']})")
            if i < len(recommendations):
                print("-" * 70)
    
    def run_analytics(self):
        """Display system analytics"""
        print("\n📊 COSMIC ANALYTICS DASHBOARD")
        print("=" * 50)
        
        df = self.astrologers_data
        
        # Basic stats
        print(f"👥 Total Astrologers: {len(df)}")
        print(f"⭐ Average Rating: {df['rating'].mean():.1f}/5.0")
        print(f"📅 Average Experience: {df['experience'].mean():.1f} years")
        print(f"🏆 Top Rated: {df.loc[df['rating'].idxmax(), 'name']}")
        print(f"🧙 Most Experienced: {df.loc[df['experience'].idxmax(), 'name']}")
        
        # Specialty distribution
        all_specialties = []
        for specialties in df['specialties']:
            all_specialties.extend(specialties)
        
        from collections import Counter
        top_specialties = Counter(all_specialties).most_common(5)
        
        print(f"\n🎯 Top Specialties:")
        for specialty, count in top_specialties:
            print(f"   • {specialty.title()}: {count} astrologers")
        
        # Focus area distribution
        focus_areas = df['focus_area'].value_counts()
        print(f"\n🌟 Focus Area Coverage:")
        for area, count in focus_areas.items():
            print(f"   • {str(area).title()}: {count} specialists")
    
    def demo_mode(self):
        """Run comprehensive demo"""
        print("\n🚀 COSMIC ENGINE DEMO MODE")
        print("=" * 50)
        
        # Test queries for recommendations
        test_queries = [
            "I need help with my love life and relationships",
            "Career guidance and business success advice", 
            "Health and wellness through astrology",
            "Family problems and marriage issues",
            "Spiritual growth and meditation guidance"
        ]
        
        print("\n🔍 RECOMMENDATION ENGINE TESTS:")
        for query in test_queries:
            recs = self.find_astrologer(query, top_k=2)
            print(f"\n💫 Query: '{query}'")
            if recs:
                for i, rec in enumerate(recs, 1):
                    emoji = self._get_relevance_emoji(rec['score_percentage'])
                    print(f"   {i}. {rec['name']} {emoji} {rec['score_percentage']}% - {rec['focus']}")
            else:
                print("   No matches found")
        
        # Test AI astrologer
        print(f"\n🤖 AI ASTROLOGER TESTS:")
        ai_tests = [
            "Your sun is in Leo and Saturn is in your 10th house",
            "I have Mars in Scorpio and need guidance",
            "What does the future hold for me?",
            "Venus in Pisces and Jupiter in my 2nd house"
        ]
        
        for test in ai_tests:
            reading = self.get_ai_reading(test)
            print(f"\n🌙 Input: '{test}'")
            print(f"🔮 Reading: {reading}")
    
    def interactive_mode(self):
        """Run interactive cosmic consultation"""
        print("\n✨ INTERACTIVE COSMIC CONSULTATION")
        print("Commands: 'find' | 'reading' | 'analytics' | 'demo' | 'quit'")
        print("=" * 60)
        
        while True:
            try:
                command = input("\n🔮 Choose your cosmic path: ").strip().lower()
                
                if command == 'quit' or command == 'q':
                    print("🌟 May the cosmic energies guide your journey ahead!")
                    break
                    
                elif command == 'find' or command == 'f':
                    query = input("💫 What guidance do you seek? ")
                    if query.strip():
                        recs = self.find_astrologer(query, top_k=3)
                        self.print_recommendations(recs, query)
                    else:
                        print("Please enter your cosmic question.")
                
                elif command == 'reading' or command == 'r':
                    astrological_input = input("🌙 Share your astrological details or question: ")
                    if astrological_input.strip():
                        reading = self.get_ai_reading(astrological_input)
                        print(f"\n🔮 YOUR COSMIC READING:")
                        print("-" * 40)
                        print(reading)
                        print("-" * 40)
                    else:
                        print("Please share your cosmic question.")
                
                elif command == 'analytics' or command == 'a':
                    self.run_analytics()
                
                elif command == 'demo' or command == 'd':
                    self.demo_mode()
                
                else:
                    print("🌟 Available commands: find | reading | analytics | demo | quit")
                    
            except KeyboardInterrupt:
                print("\n🌟 Cosmic consultation ended. Blessed be!")
                break
            except Exception as e:
                print(f"⚠️ Cosmic interference detected: {e}")

def main():
    """Main execution function"""
    print("🔮✨ COSMIC ASTROLOGER ENGINE ✨🔮")
    print("Advanced NLP Recommendation System with Percentage Relevance")
    print("Created by: DevanshSrajput | Date: 2025-07-23 18:16:14 UTC")
    print("=" * 60)
    
    # Initialize engine
    engine = CosmicAstrologerEngine()
    
    # Start interactive mode directly
    engine.interactive_mode()

if __name__ == "__main__":
    main()