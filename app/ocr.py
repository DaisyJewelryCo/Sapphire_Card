import keras_ocr
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import re
from thefuzz import process, fuzz
import tensorflow as tf
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class OCREngine:
    def __init__(self):
        """Initialize Keras-OCR pipeline."""
        print("Initializing Keras-OCR engine...")
        try:
            # Create Keras-OCR pipeline
            self.pipeline = keras_ocr.pipeline.Pipeline()
            print("✓ Keras-OCR engine initialized successfully")
        except Exception as e:
            print(f"Error initializing Keras-OCR: {e}")
            self.pipeline = None
    
    def extract_card_name(self, name_region: np.ndarray) -> str:
        """Extract card name from the name region using Keras-OCR."""
        if self.pipeline is None:
            return ""
        
        try:
            # Convert BGR to RGB if needed
            if len(name_region.shape) == 3 and name_region.shape[2] == 3:
                image_rgb = cv2.cvtColor(name_region, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = name_region
            
            # Run OCR
            predictions = self.pipeline.recognize([image_rgb])
            
            if not predictions or not predictions[0]:
                return ""
            
            # Extract text from predictions
            texts = []
            for prediction in predictions[0]:
                text, box = prediction
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
        """Extract card text from the text region using Keras-OCR."""
        if self.pipeline is None:
            return ""
        
        try:
            # Convert BGR to RGB if needed
            if len(text_region.shape) == 3 and text_region.shape[2] == 3:
                image_rgb = cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = text_region
            
            # Run OCR
            predictions = self.pipeline.recognize([image_rgb])
            
            if not predictions or not predictions[0]:
                return ""
            
            # Sort predictions by y-coordinate to maintain reading order
            sorted_predictions = self._sort_predictions_by_position(predictions[0])
            
            # Extract text from predictions
            texts = []
            for prediction in sorted_predictions:
                text, box = prediction
                texts.append(text)
            
            # Join text with appropriate spacing
            raw_text = self._reconstruct_text_layout(sorted_predictions)
            
            # Clean up the text
            cleaned = self._clean_text(raw_text, preserve_newlines=True)
            return cleaned
            
        except Exception as e:
            print(f"Error extracting card text: {e}")
            return ""
    
    def _sort_predictions_by_position(self, predictions: List[Tuple]) -> List[Tuple]:
        """Sort OCR predictions by their position (top to bottom, left to right)."""
        def get_center_y(prediction):
            text, box = prediction
            return np.mean([point[1] for point in box])
        
        def get_center_x(prediction):
            text, box = prediction
            return np.mean([point[0] for point in box])
        
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        return sorted(predictions, key=lambda p: (get_center_y(p), get_center_x(p)))
    
    def _reconstruct_text_layout(self, sorted_predictions: List[Tuple]) -> str:
        """Reconstruct text layout from sorted predictions."""
        if not sorted_predictions:
            return ""
        
        lines = []
        current_line = []
        current_y = None
        y_threshold = 20  # Pixels threshold for same line
        
        for prediction in sorted_predictions:
            text, box = prediction
            center_y = np.mean([point[1] for point in box])
            
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
        if self.pipeline is None:
            return 0.0
        
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # Run OCR
            predictions = self.pipeline.recognize([image_rgb])
            
            if not predictions or not predictions[0]:
                return 0.0
            
            # Keras-OCR doesn't provide confidence scores directly
            # We'll estimate based on the number of detected words and their length
            total_chars = sum(len(pred[0]) for pred in predictions[0])
            num_words = len(predictions[0])
            
            if num_words == 0:
                return 0.0
            
            # Simple heuristic: longer words and more words = higher confidence
            avg_word_length = total_chars / num_words
            confidence = min(100, (avg_word_length * 10) + (num_words * 5))
            
            return confidence
            
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