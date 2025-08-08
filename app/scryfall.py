import requests
import time
import json
from typing import Optional, Dict, Any
import os

class ScryfallAPI:
    def __init__(self, user_agent: str = "CardScanner/1.0"):
        self.base_url = "https://api.scryfall.com"
        self.user_agent = user_agent
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json"
        })
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Ensure we don't exceed Scryfall's rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_card_by_name(self, card_name: str) -> Optional[Dict[str, Any]]:
        """Get card data from Scryfall by name."""
        self._rate_limit()
        
        url = f"{self.base_url}/cards/named"
        params = {"fuzzy": card_name}
        
        try:
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                print(f"Card '{card_name}' not found on Scryfall")
                return None
            elif response.status_code == 429:
                print("Rate limit exceeded, waiting...")
                time.sleep(1)
                return self.get_card_by_name(card_name)  # Retry
            else:
                print(f"Scryfall API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error querying Scryfall: {e}")
            return None
    
    def parse_card_data(self, card_json: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Scryfall card JSON into a standardized format."""
        try:
            # Handle double-faced cards
            if card_json.get('card_faces'):
                name = " // ".join(face['name'] for face in card_json['card_faces'])
                # Use the first face's image
                image_url = card_json['card_faces'][0].get('image_uris', {}).get('normal', '')
                oracle_text = "\n---\n".join(face.get('oracle_text', '') for face in card_json['card_faces'])
                mana_cost = card_json['card_faces'][0].get('mana_cost', '')
                type_line = " // ".join(face.get('type_line', '') for face in card_json['card_faces'])
            else:
                name = card_json.get('name', '')
                image_url = card_json.get('image_uris', {}).get('normal', '')
                oracle_text = card_json.get('oracle_text', '')
                mana_cost = card_json.get('mana_cost', '')
                type_line = card_json.get('type_line', '')
            
            parsed_data = {
                "name": name,
                "set_code": card_json.get("set", ""),
                "set_name": card_json.get("set_name", ""),
                "collector_number": card_json.get("collector_number", ""),
                "oracle_text": oracle_text,
                "type_line": type_line,
                "mana_cost": mana_cost,
                "cmc": card_json.get("cmc", 0),
                "power": card_json.get("power"),
                "toughness": card_json.get("toughness"),
                "loyalty": card_json.get("loyalty"),
                "rarity": card_json.get("rarity", ""),
                "color_identity": card_json.get("color_identity", []),
                "colors": card_json.get("colors", []),
                "legalities": card_json.get("legalities", {}),
                "prices": card_json.get("prices", {}),
                "image_url": image_url,
                "scryfall_uri": card_json.get("scryfall_uri", ""),
                "scryfall_id": card_json.get("id", ""),
                "released_at": card_json.get("released_at", ""),
                "artist": card_json.get("artist", ""),
                "border_color": card_json.get("border_color", ""),
                "frame": card_json.get("frame", ""),
                "full_art": card_json.get("full_art", False),
                "textless": card_json.get("textless", False),
                "booster": card_json.get("booster", False),
                "story_spotlight": card_json.get("story_spotlight", False),
                "promo": card_json.get("promo", False),
                "reprint": card_json.get("reprint", False)
            }
            
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing card data: {e}")
            return {}

class PokemonAPI:
    def __init__(self):
        self.base_url = "https://api.pokemontcg.io/v2"
        self.session = requests.Session()
        # You would need an API key for production use
        # self.session.headers.update({"X-Api-Key": "your-api-key"})
    
    def get_card_by_name(self, card_name: str) -> Optional[Dict[str, Any]]:
        """Get Pokemon card data by name."""
        url = f"{self.base_url}/cards"
        params = {"q": f"name:{card_name}"}
        
        try:
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('data'):
                    return data['data'][0]  # Return first match
            
            return None
            
        except Exception as e:
            print(f"Error querying Pokemon API: {e}")
            return None
    
    def parse_card_data(self, card_json: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Pokemon card JSON into a standardized format."""
        try:
            parsed_data = {
                "name": card_json.get("name", ""),
                "set_code": card_json.get("set", {}).get("id", ""),
                "set_name": card_json.get("set", {}).get("name", ""),
                "number": card_json.get("number", ""),
                "artist": card_json.get("artist", ""),
                "rarity": card_json.get("rarity", ""),
                "hp": card_json.get("hp"),
                "types": card_json.get("types", []),
                "subtypes": card_json.get("subtypes", []),
                "supertype": card_json.get("supertype", ""),
                "attacks": card_json.get("attacks", []),
                "weaknesses": card_json.get("weaknesses", []),
                "resistances": card_json.get("resistances", []),
                "retreat_cost": card_json.get("retreatCost", []),
                "converted_retreat_cost": card_json.get("convertedRetreatCost", 0),
                "image_url": card_json.get("images", {}).get("large", ""),
                "small_image_url": card_json.get("images", {}).get("small", ""),
                "tcgplayer": card_json.get("tcgplayer", {}),
                "cardmarket": card_json.get("cardmarket", {}),
                "legalities": card_json.get("legalities", {}),
                "regulation_mark": card_json.get("regulationMark", ""),
                "released_at": card_json.get("set", {}).get("releaseDate", "")
            }
            
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing Pokemon card data: {e}")
            return {}

class CardDataManager:
    def __init__(self, cache_dir: str = "card_cache"):
        self.cache_dir = cache_dir
        self.scryfall = ScryfallAPI()
        self.pokemon = PokemonAPI()
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_card_data(self, card_name: str, card_type: str) -> Optional[Dict[str, Any]]:
        """Get card data based on card type."""
        # Check cache first
        cache_file = os.path.join(self.cache_dir, f"{card_type}_{card_name.replace(' ', '_')}.json")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cached data: {e}")
        
        # Fetch from API
        card_data = None
        
        if card_type.upper() == "MTG":
            raw_data = self.scryfall.get_card_by_name(card_name)
            if raw_data:
                card_data = self.scryfall.parse_card_data(raw_data)
                card_data["card_type"] = "MTG"
        
        elif card_type.upper() == "POKEMON":
            raw_data = self.pokemon.get_card_by_name(card_name)
            if raw_data:
                card_data = self.pokemon.parse_card_data(raw_data)
                card_data["card_type"] = "Pokemon"
        
        elif card_type.upper() == "SPORTS":
            # Sports cards would need a different API or database
            # For now, create a basic structure
            card_data = {
                "name": card_name,
                "card_type": "Sports",
                "set_name": "Unknown",
                "rarity": "Unknown",
                "image_url": "",
                "prices": {}
            }
        
        # Cache the result
        if card_data:
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(card_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"Error caching card data: {e}")
        
        return card_data
    
    def download_card_image(self, image_url: str, filename: str, output_dir: str = "card_images") -> Optional[str]:
        """Download card image from URL."""
        if not image_url:
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        try:
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return filepath
        except Exception as e:
            print(f"Error downloading image: {e}")
        
        return None