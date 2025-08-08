import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid

class DatabaseManager:
    def __init__(self, db_path: str = "card_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create cards table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cards (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                card_type TEXT NOT NULL,
                set_code TEXT,
                set_name TEXT,
                rarity TEXT,
                condition TEXT DEFAULT 'Near Mint',
                quantity INTEGER DEFAULT 1,
                estimated_value REAL,
                capture_timestamp TEXT,
                image_path TEXT,
                data_json TEXT,
                uploaded BOOLEAN DEFAULT FALSE,
                notes TEXT
            )
        ''')
        
        # Create batches table for grouping cards
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batches (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT,
                description TEXT,
                total_cards INTEGER DEFAULT 0,
                total_value REAL DEFAULT 0.0,
                uploaded BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create batch_cards junction table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batch_cards (
                batch_id TEXT,
                card_id TEXT,
                added_at TEXT,
                FOREIGN KEY (batch_id) REFERENCES batches (id),
                FOREIGN KEY (card_id) REFERENCES cards (id),
                PRIMARY KEY (batch_id, card_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_card(self, card_data: Dict[str, Any], image_path: str = "", condition: str = "Near Mint", 
                 quantity: int = 1, notes: str = "") -> str:
        """Add a card to the database."""
        card_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Extract price information
        estimated_value = 0.0
        if 'prices' in card_data:
            prices = card_data['prices']
            if isinstance(prices, dict):
                # Try different price fields
                for price_key in ['usd', 'usd_foil', 'eur', 'tix']:
                    if prices.get(price_key):
                        try:
                            estimated_value = float(prices[price_key])
                            break
                        except (ValueError, TypeError):
                            continue
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO cards (
                id, name, card_type, set_code, set_name, rarity, condition,
                quantity, estimated_value, capture_timestamp, image_path,
                data_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            card_id,
            card_data.get('name', ''),
            card_data.get('card_type', ''),
            card_data.get('set_code', ''),
            card_data.get('set_name', ''),
            card_data.get('rarity', ''),
            condition,
            quantity,
            estimated_value,
            timestamp,
            image_path,
            json.dumps(card_data),
            notes
        ))
        
        conn.commit()
        conn.close()
        
        return card_id
    
    def get_cards(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get cards from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM cards 
            ORDER BY capture_timestamp DESC 
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        
        columns = [description[0] for description in cursor.description]
        cards = []
        
        for row in cursor.fetchall():
            card_dict = dict(zip(columns, row))
            # Parse JSON data
            if card_dict['data_json']:
                try:
                    card_dict['data'] = json.loads(card_dict['data_json'])
                except json.JSONDecodeError:
                    card_dict['data'] = {}
            else:
                card_dict['data'] = {}
            
            cards.append(card_dict)
        
        conn.close()
        return cards
    
    def create_batch(self, name: str, description: str = "") -> str:
        """Create a new batch."""
        batch_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO batches (id, name, created_at, description)
            VALUES (?, ?, ?, ?)
        ''', (batch_id, name, timestamp, description))
        
        conn.commit()
        conn.close()
        
        return batch_id
    
    def add_card_to_batch(self, batch_id: str, card_id: str):
        """Add a card to a batch."""
        timestamp = datetime.now().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO batch_cards (batch_id, card_id, added_at)
            VALUES (?, ?, ?)
        ''', (batch_id, card_id, timestamp))
        
        # Update batch totals
        cursor.execute('''
            UPDATE batches SET 
                total_cards = (
                    SELECT COUNT(*) FROM batch_cards WHERE batch_id = ?
                ),
                total_value = (
                    SELECT COALESCE(SUM(c.estimated_value * c.quantity), 0)
                    FROM cards c
                    JOIN batch_cards bc ON c.id = bc.card_id
                    WHERE bc.batch_id = ?
                )
            WHERE id = ?
        ''', (batch_id, batch_id, batch_id))
        
        conn.commit()
        conn.close()
    
    def get_batches(self) -> List[Dict[str, Any]]:
        """Get all batches."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM batches ORDER BY created_at DESC')
        
        columns = [description[0] for description in cursor.description]
        batches = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return batches
    
    def get_batch_cards(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get all cards in a batch."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.*, bc.added_at as batch_added_at
            FROM cards c
            JOIN batch_cards bc ON c.id = bc.card_id
            WHERE bc.batch_id = ?
            ORDER BY bc.added_at DESC
        ''', (batch_id,))
        
        columns = [description[0] for description in cursor.description]
        cards = []
        
        for row in cursor.fetchall():
            card_dict = dict(zip(columns, row))
            if card_dict['data_json']:
                try:
                    card_dict['data'] = json.loads(card_dict['data_json'])
                except json.JSONDecodeError:
                    card_dict['data'] = {}
            else:
                card_dict['data'] = {}
            
            cards.append(card_dict)
        
        conn.close()
        return cards
    
    def mark_batch_uploaded(self, batch_id: str):
        """Mark a batch as uploaded."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE batches SET uploaded = TRUE WHERE id = ?', (batch_id,))
        cursor.execute('''
            UPDATE cards SET uploaded = TRUE 
            WHERE id IN (
                SELECT card_id FROM batch_cards WHERE batch_id = ?
            )
        ''', (batch_id,))
        
        conn.commit()
        conn.close()
    
    def delete_card(self, card_id: str):
        """Delete a card from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM batch_cards WHERE card_id = ?', (card_id,))
        cursor.execute('DELETE FROM cards WHERE id = ?', (card_id,))
        
        conn.commit()
        conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total cards
        cursor.execute('SELECT COUNT(*) FROM cards')
        total_cards = cursor.fetchone()[0]
        
        # Total value
        cursor.execute('SELECT COALESCE(SUM(estimated_value * quantity), 0) FROM cards')
        total_value = cursor.fetchone()[0]
        
        # Cards by type
        cursor.execute('SELECT card_type, COUNT(*) FROM cards GROUP BY card_type')
        cards_by_type = dict(cursor.fetchall())
        
        # Total batches
        cursor.execute('SELECT COUNT(*) FROM batches')
        total_batches = cursor.fetchone()[0]
        
        # Uploaded vs not uploaded
        cursor.execute('SELECT uploaded, COUNT(*) FROM cards GROUP BY uploaded')
        upload_status = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'total_cards': total_cards,
            'total_value': total_value,
            'cards_by_type': cards_by_type,
            'total_batches': total_batches,
            'uploaded_cards': upload_status.get(1, 0),
            'pending_cards': upload_status.get(0, 0)
        }

class ExportManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def export_batch_to_json(self, batch_id: str, output_path: str) -> bool:
        """Export a batch to JSON format."""
        try:
            batch_cards = self.db_manager.get_batch_cards(batch_id)
            
            export_data = {
                'batch_id': batch_id,
                'export_timestamp': datetime.now().isoformat(),
                'total_cards': len(batch_cards),
                'cards': []
            }
            
            for card in batch_cards:
                card_export = {
                    'id': card['id'],
                    'name': card['name'],
                    'card_type': card['card_type'],
                    'set_code': card['set_code'],
                    'set_name': card['set_name'],
                    'rarity': card['rarity'],
                    'condition': card['condition'],
                    'quantity': card['quantity'],
                    'estimated_value': card['estimated_value'],
                    'capture_timestamp': card['capture_timestamp'],
                    'image_path': card['image_path'],
                    'notes': card['notes'],
                    'data': card.get('data', {})
                }
                export_data['cards'].append(card_export)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting batch: {e}")
            return False
    
    def export_batch_to_csv(self, batch_id: str, output_path: str) -> bool:
        """Export a batch to CSV format."""
        try:
            import csv
            
            batch_cards = self.db_manager.get_batch_cards(batch_id)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'name', 'card_type', 'set_code', 'set_name', 'rarity',
                    'condition', 'quantity', 'estimated_value', 'capture_timestamp',
                    'notes'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for card in batch_cards:
                    row = {field: card.get(field, '') for field in fieldnames}
                    writer.writerow(row)
            
            return True
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.default_config = {
            "camera_index": 0,
            "ocr_confidence_threshold": 80,
            "auto_capture_enabled": False,
            "auto_capture_interval": 2.0,
            "image_save_enabled": True,
            "image_save_directory": "captured_cards",
            "cache_directory": "card_cache",
            "database_path": "card_database.db",
            "keras_ocr_models_path": "keras_ocr_models",
            "window_width": 1200,
            "window_height": 800,
            "detection_min_area": 10000,
            "detection_max_area": 500000
        }
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                merged_config = self.default_config.copy()
                merged_config.update(config)
                return merged_config
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return self.default_config.copy()
    
    def save_config(self) -> bool:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def get(self, key: str, default=None):
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a configuration value."""
        self.config[key] = value
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = self.default_config.copy()