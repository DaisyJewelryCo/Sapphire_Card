import easyocr
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import re
from thefuzz import process, fuzz
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class OCREngine:
    def __init__(self):
        """Initialize EasyOCR reader."""
        print("Initializing EasyOCR engine...")
        try:
            # Create EasyOCR reader for English
            self.reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA
            print("✓ EasyOCR engine initialized successfully")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            self.reader = None
    
    def extract_card_name(self, name_region: np.ndarray) -> str:
        """Extract card name from the name region using EasyOCR."""
        if self.reader is None:
            return ""
        
        try:
            # Run OCR
            results = self.reader.readtext(name_region)
            
            if not results:
                return ""
            
            # Extract text from results (EasyOCR returns list of (bbox, text, confidence))
            texts = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # Filter low confidence results
                    texts.append(text)
            
            # Join all detected text
            raw_text = ' '.join(texts)
            
            # Clean up the text
            cleaned = self._clean_text(raw_text)
            return cleaned
            
        except Exception as e:
            print(f"Error extracting card name: {e}")
            return ""
    
    def extract_card_text(self, text_region: np.ndarray) -> str:
        """Extract card text from the text region using EasyOCR."""
        if self.reader is None:
            return ""
        
        try:
            # Run OCR
            results = self.reader.readtext(text_region)
            
            if not results:
                return ""
            
            # Sort results by y-coordinate to maintain reading order
            sorted_results = self._sort_results_by_position(results)
            
            # Join text with appropriate spacing
            raw_text = self._reconstruct_text_layout(sorted_results)
            
            # Clean up the text
            cleaned = self._clean_text(raw_text, preserve_newlines=True)
            return cleaned
            
        except Exception as e:
            print(f"Error extracting card text: {e}")
            return ""
    
    def _sort_results_by_position(self, results: List[Tuple]) -> List[Tuple]:
        """Sort OCR results by their position (top to bottom, left to right)."""
        def get_center_y(result):
            bbox, text, confidence = result
            return np.mean([point[1] for point in bbox])
        
        def get_center_x(result):
            bbox, text, confidence = result
            return np.mean([point[0] for point in bbox])
        
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        return sorted(results, key=lambda r: (get_center_y(r), get_center_x(r)))
    
    def _reconstruct_text_layout(self, sorted_results: List[Tuple]) -> str:
        """Reconstruct text layout from sorted results."""
        if not sorted_results:
            return ""
        
        lines = []
        current_line = []
        current_y = None
        y_threshold = 20  # Pixels threshold for same line
        
        for result in sorted_results:
            bbox, text, confidence = result
            if confidence < 0.5:  # Skip low confidence results
                continue
                
            center_y = np.mean([point[1] for point in bbox])
            
            if current_y is None or abs(center_y - current_y) <= y_threshold:
                # Same line
                current_line.append(text)
                current_y = center_y if current_y is None else (current_y + center_y) / 2
            else:
                # New line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [text]
                current_y = center_y
        
        # Add the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _clean_text(self, text: str, preserve_newlines: bool = False) -> str:
        """Clean up OCR text output."""
        if not text:
            return ""
        
        # Remove extra whitespace
        if preserve_newlines:
            lines = text.split('\n')
            cleaned_lines = [' '.join(line.split()) for line in lines]
            cleaned = '\n'.join(line for line in cleaned_lines if line.strip())
        else:
            cleaned = ' '.join(text.split())
        
        # Remove non-printable characters except newlines if preserved
        if preserve_newlines:
            cleaned = ''.join(char for char in cleaned if char.isprintable() or char == '\n')
        else:
            cleaned = ''.join(char for char in cleaned if char.isprintable())
        
        return cleaned.strip()
    
    def get_text_confidence(self, image: np.ndarray) -> float:
        """Get confidence score for OCR results."""
        if self.reader is None:
            return 0.0
        
        try:
            # Run OCR
            results = self.reader.readtext(image)
            
            if not results:
                return 0.0
            
            # Calculate average confidence from EasyOCR results
            confidences = [confidence for (bbox, text, confidence) in results]
            
            if not confidences:
                return 0.0
            
            # Return average confidence as percentage
            avg_confidence = sum(confidences) / len(confidences)
            return avg_confidence * 100
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.0

class CardMatcher:
    def __init__(self):
        self.mtg_cards = []
        self.pokemon_cards = []
        self.sports_cards = []
        self.all_cards = []
        
        # Load card databases
        self._load_card_databases()
    
    def _load_card_databases(self):
        """Load card name databases for different card types."""
        # This would typically load from files or APIs
        # For now, we'll use some example cards
        
        # MTG cards (sample)
        self.mtg_cards = [
            "Lightning Bolt", "Black Lotus", "Ancestral Recall", "Time Walk",
            "Mox Pearl", "Mox Sapphire", "Mox Jet", "Mox Ruby", "Mox Emerald",
            "Sol Ring", "Counterspell", "Dark Ritual", "Giant Growth",
            "Swords to Plowshares", "Path to Exile", "Brainstorm", "Ponder"
        ]
        
        # Pokemon cards (sample)
        self.pokemon_cards = [
            "Pikachu", "Charizard", "Blastoise", "Venusaur", "Mewtwo",
            "Mew", "Lugia", "Ho-Oh", "Rayquaza", "Arceus", "Dialga",
            "Palkia", "Giratina", "Reshiram", "Zekrom", "Kyurem"
        ]
        
        # Sports cards (sample)
        self.sports_cards = [
            "Michael Jordan", "LeBron James", "Kobe Bryant", "Magic Johnson",
            "Larry Bird", "Shaquille O'Neal", "Tim Duncan", "Kareem Abdul-Jabbar",
            "Tom Brady", "Joe Montana", "Jerry Rice", "Jim Brown",
            "Babe Ruth", "Mickey Mantle", "Ted Williams", "Lou Gehrig"
        ]
        
        self.all_cards = self.mtg_cards + self.pokemon_cards + self.sports_cards
    
    def match_card_name(self, raw_name: str, threshold: int = 80) -> Optional[Dict]:
        """Match raw OCR text to known card names."""
        if not raw_name or len(raw_name.strip()) < 2:
            return None
        
        # Try exact match first
        for card_list, card_type in [
            (self.mtg_cards, "MTG"),
            (self.pokemon_cards, "Pokemon"),
            (self.sports_cards, "Sports")
        ]:
            if raw_name in card_list:
                return {
                    "name": raw_name,
                    "type": card_type,
                    "confidence": 100,
                    "method": "exact"
                }
        
        # Try fuzzy matching
        best_match = None
        best_score = 0
        best_type = None
        
        for card_list, card_type in [
            (self.mtg_cards, "MTG"),
            (self.pokemon_cards, "Pokemon"),
            (self.sports_cards, "Sports")
        ]:
            if card_list:
                match, score = process.extractOne(raw_name, card_list)
                if score > best_score and score >= threshold:
                    best_match = match
                    best_score = score
                    best_type = card_type
        
        if best_match:
            return {
                "name": best_match,
                "type": best_type,
                "confidence": best_score,
                "method": "fuzzy"
            }
        
        return None
    
    def detect_card_type(self, card_text: str) -> str:
        """Detect card type based on text content."""
        text_lower = card_text.lower()
        
        # MTG keywords
        mtg_keywords = [
            "mana", "tap", "untap", "creature", "instant", "sorcery",
            "enchantment", "artifact", "planeswalker", "land", "library",
            "graveyard", "battlefield", "exile", "cast", "spell"
        ]
        
        # Pokemon keywords
        pokemon_keywords = [
            "pokemon", "energy", "trainer", "hp", "attack", "weakness",
            "resistance", "retreat", "evolves", "basic", "stage"
        ]
        
        # Sports keywords
        sports_keywords = [
            "rookie", "draft", "season", "team", "player", "stats",
            "championship", "mvp", "all-star", "hall of fame"
        ]
        
        mtg_score = sum(1 for keyword in mtg_keywords if keyword in text_lower)
        pokemon_score = sum(1 for keyword in pokemon_keywords if keyword in text_lower)
        sports_score = sum(1 for keyword in sports_keywords if keyword in text_lower)
        
        if mtg_score > pokemon_score and mtg_score > sports_score:
            return "MTG"
        elif pokemon_score > sports_score:
            return "Pokemon"
        elif sports_score > 0:
            return "Sports"
        else:
            return "Unknown"