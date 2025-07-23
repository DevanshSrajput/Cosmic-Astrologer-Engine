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
        
        print("✨ Cosmic Engine Ready!")
        
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
        # Create searchable text for each astrologer with enhanced weighting
        self.astrologer_texts = []
        for _, astrologer in self.astrologers_data.iterrows():
            # Give extra weight to specialties by repeating them
            specialty_text = " ".join(astrologer['specialties']) * 3  # Triple weight
            focus_text = astrologer['focus_area'] * 2  # Double weight
            description_text = astrologer['description']
            
            # Combine all text with weighted importance
            text = f"{specialty_text} {focus_text} {description_text}"
            self.astrologer_texts.append(text.lower())
        
        # Initialize TF-IDF with optimized parameters for better matching
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for better phrase matching
            max_features=2000,   # Increased features for better discrimination
            lowercase=True,
            min_df=1,           # Don't ignore rare words
            max_df=0.95         # Ignore very common words
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.astrologer_texts)
        print("🚀 Recommendation engine optimized with enhanced TF-IDF")
    
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
        """Convert cosine similarity to percentage relevance with very aggressive scoring"""
        if score <= 0:
            return 0
        
        # Ultra-aggressive scoring to ensure high percentages for any positive match
        # Most TF-IDF cosine similarities are between 0.01-0.3 for real matches
        
        if score >= 0.3:  # Excellent match
            percentage = min(100, int(90 + (score - 0.3) * 14))  # 90-100%
        elif score >= 0.2:  # Very good match
            percentage = int(80 + (score - 0.2) * 100)  # 80-90%
        elif score >= 0.12:  # Good match
            percentage = int(65 + (score - 0.12) * 187)  # 65-80%
        elif score >= 0.08:  # Decent match
            percentage = int(50 + (score - 0.08) * 375)  # 50-65%
        elif score >= 0.05:  # Fair match
            percentage = int(35 + (score - 0.05) * 500)  # 35-50%
        elif score >= 0.03:  # Weak match
            percentage = int(25 + (score - 0.03) * 500)  # 25-35%
        elif score >= 0.01:  # Very weak match
            percentage = int(15 + (score - 0.01) * 500)  # 15-25%
        else:  # Minimal match
            percentage = max(10, int(score * 1500))  # 10-15%
        
        return min(100, percentage)
    
    def find_astrologer(self, query: str, top_k: int = 3) -> List[Dict]:
        """Find best matching astrologers using optimized search with query expansion and keyword boosting"""
        if not query.strip():
            return []
        
        # Expand query with synonyms for better matching
        expanded_query = self._expand_query(query.lower())
        
        # Transform query and calculate similarities
        query_vector = self.vectorizer.transform([expanded_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Apply keyword boosting for direct matches
        similarities = self._apply_keyword_boosting(query.lower(), similarities)
        
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
    
    def _apply_keyword_boosting(self, query: str, similarities: np.ndarray) -> np.ndarray:
        """Apply boosting for direct keyword matches"""
        boosted_similarities = similarities.copy()
        
        # Define keyword groups with their boost factors
        keyword_boosts = {
            'love': ['love', 'romance', 'romantic', 'relationship', 'dating', 'heart', 'soulmate'],
            'career': ['career', 'job', 'work', 'business', 'professional', 'money', 'success'],
            'health': ['health', 'wellness', 'healing', 'medical', 'vitality'],
            'family': ['family', 'marriage', 'children', 'home', 'parents'],
            'spiritual': ['spiritual', 'meditation', 'soul', 'enlightenment', 'karma'],
            'travel': ['travel', 'journey', 'foreign', 'adventure', 'moving'],
            'general': ['future', 'prediction', 'guidance', 'destiny']
        }
        
        # Check each astrologer for keyword matches and apply boosts
        for idx, astrologer in self.astrologers_data.iterrows():
            focus_area = astrologer['focus_area']
            specialties = astrologer['specialties']
            
            boost_factor = 1.0
            
            # Check if query contains keywords matching this astrologer's focus
            if focus_area in keyword_boosts:
                matching_keywords = keyword_boosts[focus_area]
                for keyword in matching_keywords:
                    if keyword in query:
                        boost_factor += 0.5  # Add 50% boost per matching keyword
            
            # Additional boost for direct specialty matches
            for specialty in specialties:
                if specialty in query:
                    boost_factor += 0.3  # Add 30% boost per specialty match
            
            # Apply the boost
            boosted_similarities[idx] *= boost_factor
        
        return boosted_similarities
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms for better matching"""
        # Define synonym groups for better matching
        synonyms = {
            'love': ['love', 'romance', 'romantic', 'relationship', 'dating', 'heart', 'affection'],
            'relationship': ['relationship', 'love', 'romance', 'partner', 'dating', 'marriage'],
            'career': ['career', 'job', 'work', 'professional', 'business', 'employment'],
            'health': ['health', 'wellness', 'healing', 'medical', 'vitality', 'well-being'],
            'family': ['family', 'relatives', 'parents', 'children', 'home', 'domestic'],
            'spiritual': ['spiritual', 'soul', 'meditation', 'enlightenment', 'chakra'],
            'money': ['money', 'finance', 'financial', 'wealth', 'prosperity', 'abundance'],
            'future': ['future', 'prediction', 'forecast', 'destiny', 'fate', 'tomorrow']
        }
        
        # Expand the query with synonyms
        expanded_terms = [query]
        query_words = query.split()
        
        for word in query_words:
            for key, synonym_list in synonyms.items():
                if word in synonym_list:
                    expanded_terms.extend(synonym_list)
                    break
        
        # Remove duplicates and join
        unique_terms = list(set(expanded_terms))
        return ' '.join(unique_terms)
    
    def get_ai_reading(self, input_text: str) -> str:
        """Generate AI astrologer reading with context-aware responses"""
        if not input_text.strip():
            return "Please share your cosmic question for guidance."
        
        input_lower = input_text.lower()
        reading_parts = []
        
        # Analyze question context for personalized response
        question_context = self._analyze_question_context(input_lower)
        
        # Context-aware mystical opening
        opening = self._get_contextual_opening(question_context)
        reading_parts.append(opening)
        
        # Parse astrological elements
        astrological_elements = self._parse_astrological_elements(input_lower)
        
        # Generate context-specific insights
        context_insights = self._generate_context_insights(question_context, input_lower)
        if context_insights:
            reading_parts.extend(context_insights)
        
        # Add astrological element insights
        if astrological_elements:
            reading_parts.extend(astrological_elements)
        
        # Add contextual guidance
        guidance = self._get_contextual_guidance(question_context)
        reading_parts.append(guidance)
        
        # Add specific predictions or advice based on context
        predictions = self._generate_predictions(question_context, input_lower)
        if predictions:
            reading_parts.extend(predictions)
        
        # Context-appropriate blessing
        blessing = self._get_contextual_blessing(question_context)
        reading_parts.append(blessing)
        
        return " ".join(reading_parts)
    
    def _analyze_question_context(self, input_text: str) -> str:
        """Analyze the context and intent of the user's question"""
        # Define context keywords
        contexts = {
            'love': ['love', 'relationship', 'romance', 'dating', 'marriage', 'partner', 'soulmate', 'heart', 'crush', 'boyfriend', 'girlfriend', 'husband', 'wife'],
            'career': ['job', 'career', 'work', 'business', 'money', 'success', 'promotion', 'professional', 'employment', 'salary', 'finance'],
            'health': ['health', 'illness', 'healing', 'medical', 'wellness', 'body', 'mental', 'stress', 'anxiety', 'depression', 'recovery'],
            'family': ['family', 'parents', 'children', 'mother', 'father', 'siblings', 'home', 'domestic', 'relatives'],
            'spiritual': ['spiritual', 'soul', 'meditation', 'enlightenment', 'karma', 'past life', 'chakra', 'energy', 'awakening'],
            'future': ['future', 'prediction', 'what will', 'when will', 'tomorrow', 'next year', 'upcoming', 'forecast'],
            'travel': ['travel', 'journey', 'move', 'relocation', 'foreign', 'abroad', 'trip', 'adventure'],
            'general': ['life', 'purpose', 'guidance', 'help', 'advice', 'direction', 'path']
        }
        
        # Count matches for each context
        context_scores = {}
        for context, keywords in contexts.items():
            score = sum(1 for keyword in keywords if keyword in input_text)
            if score > 0:
                context_scores[context] = score
        
        # Return the context with highest score, or 'general' if no matches
        if context_scores:
            return max(context_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def _get_contextual_opening(self, context: str) -> str:
        """Get context-specific opening based on question type"""
        openings = {
            'love': [
                "The celestial energies of Venus reveal profound insights about your heart's journey.",
                "The cosmic dance of love surrounds you with powerful romantic vibrations.",
                "Your heart chakra glows with divine light as the universe prepares romantic blessings."
            ],
            'career': [
                "Jupiter's expansive energy illuminates your professional path with golden opportunities.",
                "The cosmic wheels of success are turning in your favor, bringing career advancement.",
                "Saturn's disciplined energy guides your ambitions toward material achievement."
            ],
            'health': [
                "The healing energies of the universe flow through your being, restoring balance.",
                "Your life force energy is realigning with cosmic harmony for optimal wellness.",
                "The celestial healers whisper remedies for your physical and spiritual well-being."
            ],
            'family': [
                "The protective energies of Cancer surround your family with nurturing love.",
                "Ancestral wisdom flows through your bloodline, bringing family harmony.",
                "The fourth house of home activates with divine blessings for your loved ones."
            ],
            'spiritual': [
                "Your soul's ancient wisdom awakens as you walk the path of enlightenment.",
                "The veil between dimensions grows thin, revealing sacred spiritual truths.",
                "Your higher self communicates through cosmic frequencies of divine love."
            ],
            'future': [
                "The crystal ball of time reveals glimpses of your destined future path.",
                "Prophectic visions dance across the cosmic canvas, showing what's to come.",
                "The threads of fate weave together to reveal your upcoming life chapters."
            ],
            'travel': [
                "Mercury's swift energy opens pathways to distant lands and new adventures.",
                "The cosmic compass points toward transformative journeys awaiting you.",
                "Wanderlust energies activate as the universe calls you to explore new horizons."
            ],
            'general': [
                "The cosmic energies reveal", "Your celestial blueprint shows", "The stars whisper",
                "Ancient wisdom flows through", "Divine timing brings", "The universe aligns"
            ]
        }
        
        context_openings = openings.get(context, openings['general'])
        return random.choice(context_openings)
    
    def _parse_astrological_elements(self, input_text: str) -> List[str]:
        """Parse and respond to specific astrological elements mentioned"""
        elements = []
        
        # Check for zodiac signs
        for sign, traits in self.zodiac_traits.items():
            if sign in input_text:
                trait = random.choice(traits)
                elements.append(f"Your {sign.title()} energy emphasizes {trait}, creating powerful manifestations.")
        
        # Check for planets
        for planet, meaning in self.planets.items():
            if planet in input_text:
                elements.append(f"{planet.title()}'s influence on your {meaning} brings transformative changes.")
        
        # Check for houses
        house_pattern = r'(\d+)(?:st|nd|rd|th)?\s*house'
        houses = re.findall(house_pattern, input_text)
        for house_num in houses:
            if house_num in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
                house_key = f"{house_num}{'st' if house_num == '1' else 'nd' if house_num == '2' else 'rd' if house_num == '3' else 'th'}"
                if house_key in self.house_meanings:
                    meaning = self.house_meanings[house_key]
                    elements.append(f"The {house_key} house of {meaning} awakens with cosmic significance.")
        
        return elements
    
    def _generate_context_insights(self, context: str, input_text: str) -> List[str]:
        """Generate specific insights based on question context"""
        insights = {
            'love': [
                "A significant romantic connection approaches your energy field.",
                "Your heart is ready to receive the love you truly deserve.",
                "Past relationship patterns are healing, making space for true love."
            ],
            'career': [
                "Professional recognition comes through showcasing your unique talents.",
                "A new opportunity will test your skills and reward your dedication.",
                "Financial abundance flows as you align with your true calling."
            ],
            'health': [
                "Your body's natural healing mechanisms are activating powerfully.",
                "Energy blockages are clearing, restoring your vitality and strength.",
                "Mind-body-spirit alignment brings optimal wellness and peace."
            ],
            'family': [
                "Family bonds strengthen through understanding and compassion.",
                "A family situation requires your wisdom and gentle guidance.",
                "Generational healing flows through your family lineage now."
            ],
            'spiritual': [
                "Your psychic abilities are expanding with cosmic awakening.",
                "Past life memories surface to provide current life guidance.",
                "Your spirit guides send clear messages through synchronicities."
            ],
            'future': [
                "The next three months bring significant positive changes.",
                "A door that seemed closed will unexpectedly open wide.",
                "Your patience will be rewarded with exactly what you've hoped for."
            ],
            'travel': [
                "A journey, physical or spiritual, will transform your perspective.",
                "Foreign connections bring unexpected opportunities and growth.",
                "Your next adventure holds keys to your personal evolution."
            ],
            'general': [
                "The cosmic currents shift to support your highest good.",
                "Divine timing orchestrates events perfectly for your growth.",
                "Your soul's purpose becomes clearer with each passing day."
            ]
        }
        
        context_insights = insights.get(context, insights['general'])
        return [random.choice(context_insights)]
    
    def _get_contextual_guidance(self, context: str) -> str:
        """Provide context-specific guidance"""
        guidance = {
            'love': [
                "Open your heart to receive love while maintaining healthy boundaries.",
                "Trust your intuition when making decisions about relationships.",
                "Self-love is the foundation for attracting your ideal partner."
            ],
            'career': [
                "Take calculated risks that align with your long-term vision.",
                "Network authentically and let your genuine self shine through.",
                "Persistence combined with flexibility will yield the best results."
            ],
            'health': [
                "Listen to your body's wisdom and honor its need for rest.",
                "Incorporate meditation and mindful breathing into your daily routine.",
                "Seek balance between activity and restoration for optimal health."
            ],
            'family': [
                "Practice patience and compassion in all family interactions.",
                "Set loving boundaries to protect your energy and peace.",
                "Focus on understanding rather than being understood."
            ],
            'spiritual': [
                "Trust the spiritual downloads and insights you're receiving.",
                "Create sacred space for daily meditation and reflection.",
                "Pay attention to recurring dreams and symbolic messages."
            ],
            'future': [
                "Stay present while remaining open to emerging possibilities.",
                "Trust that everything is unfolding in divine perfect timing.",
                "Prepare yourself energetically for the blessings coming your way."
            ],
            'travel': [
                "Research thoroughly but remain flexible in your travel plans.",
                "Pack light in possessions but heavy in positive intentions.",
                "Every journey teaches valuable lessons about yourself."
            ],
            'general': [
                "Trust your intuition during this powerful time of change.",
                "Embrace the transformative energies surrounding you.",
                "Stay grounded while reaching for your highest aspirations."
            ]
        }
        
        context_guidance = guidance.get(context, guidance['general'])
        return random.choice(context_guidance)
    
    def _generate_predictions(self, context: str, input_text: str) -> List[str]:
        """Generate specific predictions based on context"""
        # Only generate predictions for future-oriented questions
        future_keywords = ['when', 'will', 'future', 'next', 'soon', 'coming', 'ahead']
        if not any(keyword in input_text for keyword in future_keywords):
            return []
        
        predictions = {
            'love': [
                "Within the next lunar cycle, a meaningful connection enters your life.",
                "By the autumn equinox, clarity about a relationship situation emerges."
            ],
            'career': [
                "Before the year's end, a significant professional opportunity manifests.",
                "The next Mercury transit brings important career communications."
            ],
            'health': [
                "Your energy levels will notably improve within the coming month.",
                "A breakthrough in your wellness journey occurs during the next season."
            ],
            'family': [
                "A family celebration or reunion brings joy in the near future.",
                "Resolution to a family matter comes through open communication soon."
            ],
            'spiritual': [
                "A spiritual awakening accelerates during the next full moon cycle.",
                "Your meditation practice deepens significantly in the coming weeks."
            ],
            'general': [
                "A positive shift in your life circumstances occurs within 90 days.",
                "The seeds you're planting now will bloom beautifully by next season."
            ]
        }
        
        context_predictions = predictions.get(context, predictions['general'])
        return [random.choice(context_predictions)]
    
    def _get_contextual_blessing(self, context: str) -> str:
        """Get context-appropriate blessing"""
        blessings = {
            'love': [
                "May your heart overflow with the love you seek and deserve.",
                "May divine love surround you and guide all your relationships.",
                "May you attract a love that honors and celebrates your true self."
            ],
            'career': [
                "May success flow to you through channels of integrity and purpose.",
                "May your work bring both material abundance and soul satisfaction.",
                "May your professional path align perfectly with your highest calling."
            ],
            'health': [
                "May vibrant health and energy be your constant companions.",
                "May your body temple be blessed with strength and vitality.",
                "May healing light restore perfect balance to your entire being."
            ],
            'family': [
                "May your family bonds be strengthened with love and understanding.",
                "May peace and harmony reign in your home and heart.",
                "May generations of wisdom flow through your family lineage."
            ],
            'spiritual': [
                "May your spiritual journey be filled with wonder and enlightenment.",
                "May divine guidance illuminate every step of your sacred path.",
                "May you embody the highest version of your soul's expression."
            ],
            'future': [
                "May your future unfold with grace, joy, and divine perfection.",
                "May all your dreams manifest in ways that exceed your expectations.",
                "May the path ahead be filled with blessings and beautiful surprises."
            ],
            'travel': [
                "May your journeys be safe, transformative, and filled with wonder.",
                "May every path you walk lead to greater wisdom and joy.",
                "May adventures near and far expand your heart and consciousness."
            ],
            'general': [
                "May the stars illuminate your path ahead with brilliant light.",
                "May divine blessings flow abundantly into every area of your life.",
                "May you walk in harmony with your highest purpose and deepest joy."
            ]
        }
        
        context_blessings = blessings.get(context, blessings['general'])
        return random.choice(context_blessings)
    
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
    
    def show_main_menu(self):
        """Display main menu and get user's initial path choice"""
        print("\n🔮 COSMIC ASTROLOGER ENGINE - MAIN MENU")
        print("=" * 50)
        print("Choose your cosmic journey:")
        print("1. 🔍 Find Perfect Astrologer")
        print("2. 🤖 Get AI Astrologer Reading") 
        print("3. 📊 Run Demo & Analytics")
        print("4. ✨ Interactive Mode (All Features)")
        print("5. 🚪 Exit")
        print("=" * 50)
        
        while True:
            try:
                choice = input("\n🌟 Enter your choice (1-5): ").strip()
                
                if choice == '1':
                    self.astrologer_finder_mode()
                    break
                elif choice == '2':
                    self.ai_reading_mode()
                    break
                elif choice == '3':
                    self.demo_and_analytics_mode()
                    break
                elif choice == '4':
                    self.interactive_mode()
                    break
                elif choice == '5':
                    print("🌟 May the cosmic energies guide your journey ahead!")
                    break
                else:
                    print("⚠️ Please enter a valid choice (1-5)")
                    
            except KeyboardInterrupt:
                print("\n🌟 Cosmic consultation ended. Blessed be!")
                break
            except Exception as e:
                print(f"⚠️ Cosmic interference detected: {e}")
    
    def astrologer_finder_mode(self):
        """Dedicated mode for finding astrologers"""
        print("\n🔍 ASTROLOGER FINDER MODE")
        print("=" * 40)
        print("Find the perfect astrologer for your needs!")
        print("Type 'back' to return to main menu")
        
        while True:
            try:
                query = input("\n💫 What guidance do you seek? ").strip()
                
                if query.lower() == 'back':
                    self.show_main_menu()
                    break
                elif query:
                    recs = self.find_astrologer(query, top_k=3)
                    self.print_recommendations(recs, query)
                    
                    continue_choice = input("\n🌟 Search again? (y/n): ").strip().lower()
                    if continue_choice != 'y':
                        self.show_main_menu()
                        break
                else:
                    print("Please enter your cosmic question.")
                    
            except KeyboardInterrupt:
                print("\n🌟 Returning to main menu...")
                self.show_main_menu()
                break
            except Exception as e:
                print(f"⚠️ Cosmic interference detected: {e}")
    
    def ai_reading_mode(self):
        """Dedicated mode for AI readings"""
        print("\n🤖 AI ASTROLOGER READING MODE")
        print("=" * 40)
        print("Get personalized cosmic insights from our AI astrologer!")
        print("Type 'back' to return to main menu")
        
        while True:
            try:
                astrological_input = input("\n🌙 Share your astrological details or question: ").strip()
                
                if astrological_input.lower() == 'back':
                    self.show_main_menu()
                    break
                elif astrological_input:
                    reading = self.get_ai_reading(astrological_input)
                    print(f"\n🔮 YOUR COSMIC READING:")
                    print("-" * 40)
                    print(reading)
                    print("-" * 40)
                    
                    continue_choice = input("\n🌟 Get another reading? (y/n): ").strip().lower()
                    if continue_choice != 'y':
                        self.show_main_menu()
                        break
                else:
                    print("Please share your cosmic question.")
                    
            except KeyboardInterrupt:
                print("\n🌟 Returning to main menu...")
                self.show_main_menu()
                break
            except Exception as e:
                print(f"⚠️ Cosmic interference detected: {e}")
    
    def demo_and_analytics_mode(self):
        """Dedicated mode for demo and analytics"""
        print("\n📊 DEMO & ANALYTICS MODE")
        print("=" * 40)
        print("Explore system capabilities and view analytics!")
        print("1. 🚀 Run Demo")
        print("2. 📊 View Analytics") 
        print("3. 🔙 Back to Main Menu")
        
        while True:
            try:
                choice = input("\n🌟 Choose option (1-3): ").strip()
                
                if choice == '1':
                    self.demo_mode()
                elif choice == '2':
                    self.run_analytics()
                elif choice == '3':
                    self.show_main_menu()
                    break
                else:
                    print("⚠️ Please enter a valid choice (1-3)")
                    
                continue_choice = input("\n🌟 Continue in this mode? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    self.show_main_menu()
                    break
                    
            except KeyboardInterrupt:
                print("\n🌟 Returning to main menu...")
                self.show_main_menu()
                break
            except Exception as e:
                print(f"⚠️ Cosmic interference detected: {e}")

    def interactive_mode(self):
        """Run interactive cosmic consultation"""
        print("\n✨ INTERACTIVE COSMIC CONSULTATION")
        print("Commands: 'find' | 'reading' | 'analytics' | 'demo' | 'back' | 'quit'")
        print("=" * 60)
        
        while True:
            try:
                command = input("\n🔮 Choose your cosmic path: ").strip().lower()
                
                if command == 'quit' or command == 'q':
                    print("🌟 May the cosmic energies guide your journey ahead!")
                    break
                    
                elif command == 'back' or command == 'b':
                    self.show_main_menu()
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
                    print("🌟 Available commands: find | reading | analytics | demo | back | quit")
                    
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
    
    # Show main menu for path selection
    engine.show_main_menu()

if __name__ == "__main__":
    main()