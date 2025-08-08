from paddleocr import PaddleOCR
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
import re
from thefuzz import process, fuzz
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class OCREngine:
    def __init__(self):
        """Initialize PaddleOCR reader."""
        print("Initializing PaddleOCR engine...")
        try:
            # Import setuptools first to avoid issues
            import setuptools
            # Create PaddleOCR reader for English
            self.ocr = PaddleOCR(use_textline_orientation=True, lang='en')
            print("✓ PaddleOCR engine initialized successfully")
        except ImportError as e:
            print(f"Import error initializing PaddleOCR: {e}")
            print("Falling back to basic OCR functionality...")
            self.ocr = None
        except Exception as e:
            print(f"Error initializing PaddleOCR: {e}")
            print("Falling back to basic OCR functionality...")
            self.ocr = None
    
    def extract_card_name(self, name_region: np.ndarray) -> str:
        """Extract card name from the name region using PaddleOCR."""
        if self.ocr is None:
            print("OCR engine not available, using fallback")
            return self._basic_ocr_fallback(name_region)
        
        try:
            # Run OCR
            print("Running OCR on name region...")
            results = self.ocr.predict(name_region)
            print(f"OCR results: {results}")
            
            if not results or not results[0]:
                print("No OCR results found")
                return ""
            
            # Extract text from results (PaddleOCR returns list of [bbox, (text, confidence)])
            texts = []
            for line in results[0]:
                if line and len(line) >= 2:
                    bbox, (text, confidence) = line
                    print(f"OCR detected: '{text}' with confidence {confidence}")
                    if confidence > 0.3:  # Lower threshold for debugging
                        texts.append(text)
            
            # Join all detected text
            raw_text = ' '.join(texts)
            print(f"Raw OCR text: '{raw_text}'")
            
            # Clean up the text
            cleaned = self._clean_text(raw_text)
            print(f"Cleaned text: '{cleaned}'")
            return cleaned
            
        except Exception as e:
            print(f"Error extracting card name: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def extract_card_text(self, text_region: np.ndarray) -> str:
        """Extract card text from the text region using PaddleOCR."""
        if self.ocr is None:
            return self._basic_ocr_fallback(text_region)
        
        try:
            # Run OCR
            results = self.ocr.predict(text_region)
            
            if not results or not results[0]:
                return ""
            
            # Sort results by y-coordinate to maintain reading order
            sorted_results = self._sort_results_by_position(results[0])
            
            # Join text with appropriate spacing
            raw_text = self._reconstruct_text_layout(sorted_results)
            
            # Clean up the text
            cleaned = self._clean_text(raw_text, preserve_newlines=True)
            return cleaned
            
        except Exception as e:
            print(f"Error extracting card text: {e}")
            return ""
    
    def _sort_results_by_position(self, results: List) -> List:
        """Sort OCR results by their position (top to bottom, left to right)."""
        def get_center_y(result):
            if result and len(result) >= 2:
                bbox, (text, confidence) = result
                return np.mean([point[1] for point in bbox])
            return 0
        
        def get_center_x(result):
            if result and len(result) >= 2:
                bbox, (text, confidence) = result
                return np.mean([point[0] for point in bbox])
            return 0
        
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        return sorted(results, key=lambda r: (get_center_y(r), get_center_x(r)))
    
    def _reconstruct_text_layout(self, sorted_results: List) -> str:
        """Reconstruct text layout from sorted results."""
        if not sorted_results:
            return ""
        
        lines = []
        current_line = []
        current_y = None
        y_threshold = 20  # Pixels threshold for same line
        
        for result in sorted_results:
            if not result or len(result) < 2:
                continue
                
            bbox, (text, confidence) = result
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
    
    def _basic_ocr_fallback(self, image: np.ndarray) -> str:
        """Basic OCR fallback using OpenCV text detection."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply some basic preprocessing
            # Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # Apply threshold to get binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Try to detect text regions using contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be text
            text_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by aspect ratio and size
                if w > 10 and h > 5 and w/h > 1.5 and w/h < 10:
                    text_contours.append((x, y, w, h))
            
            # Sort by y-coordinate (top to bottom)
            text_contours.sort(key=lambda x: x[1])
            
            # For now, return a placeholder since we don't have actual OCR
            # In a real implementation, you might use pytesseract here
            if text_contours:
                return "Text detected (OCR engine not available)"
            else:
                return ""
                
        except Exception as e:
            print(f"Error in basic OCR fallback: {e}")
            return ""
    
    def get_text_confidence(self, image: np.ndarray) -> float:
        """Get confidence score for OCR results."""
        if self.ocr is None:
            return 0.0
        
        try:
            # Run OCR
            results = self.ocr.predict(image)
            
            if not results or not results[0]:
                return 0.0
            
            # Calculate average confidence from PaddleOCR results
            confidences = []
            for line in results[0]:
                if line and len(line) >= 2:
                    bbox, (text, confidence) = line
                    confidences.append(confidence)
            
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
            "Swords to Plowshares", "Path to Exile", "Brainstorm", "Ponder",
            "Forest", "Island", "Mountain", "Plains", "Swamp",
            "Llanowar Elves", "Birds of Paradise", "Serra Angel", "Shivan Dragon",
            "Lightning Strike", "Shock", "Fireball", "Disenchant", "Terror",
            "Wrath of God", "Day of Judgment", "Doom Blade", "Cancel",
            "Divination", "Opt", "Preordain", "Serum Visions", "Thoughtseize",
            "Duress", "Inquisition of Kozilek", "Fatal Push", "Bolt", "Counterspell"
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
    
    def match_card_name(self, raw_name: str, threshold: int = 60) -> Optional[Dict]:
        """Match raw OCR text to known card names."""
        print(f"Attempting to match card name: '{raw_name}'")
        
        if not raw_name or len(raw_name.strip()) < 2:
            print("Card name too short or empty")
            return None
        
        # Try exact match first
        for card_list, card_type in [
            (self.mtg_cards, "MTG"),
            (self.pokemon_cards, "Pokemon"),
            (self.sports_cards, "Sports")
        ]:
            if raw_name in card_list:
                print(f"Exact match found: {raw_name} ({card_type})")
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
                print(f"Best {card_type} match: '{match}' with score {score}")
                if score > best_score and score >= threshold:
                    best_match = match
                    best_score = score
                    best_type = card_type
        
        if best_match:
            print(f"Fuzzy match found: '{best_match}' ({best_type}) with score {best_score}")
            return {
                "name": best_match,
                "type": best_type,
                "confidence": best_score,
                "method": "fuzzy"
            }
        
        print(f"No match found for '{raw_name}' (best score was {best_score}, threshold is {threshold})")
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