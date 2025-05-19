import os
import csv
import sqlite3
import re
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DataQuerySystem:
    def __init__(self, db_name="products.db"):
        self.db_name = db_name
        self.conn = None
        self.cursor = None
        self.llm_model = None
        self.tokenizer = None
        self.queries_generated = []
        
        try:
            self._init_sqlite()
            self._init_llm("deepseek_model")
            print("System initialized successfully")
        except Exception as e:
            print(f"Initialization failed: {e}")
            raise

    def _init_sqlite(self):
        """Initialize SQLite connection"""
        try:
            self.conn = sqlite3.connect(self.db_name)
            self.cursor = self.conn.cursor()
            print("Connected to SQLite successfully")
        except Exception as e:
            print(f"Error connecting to SQLite: {e}")
            raise
    
    def _init_llm(self, model_path):
        """Initialize the DeepSeek model from local path"""
        try:
            print(f"Loading DeepSeek model from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("DeepSeek model loaded successfully")
        except Exception as e:
            print(f"Error loading DeepSeek model: {e}")
            raise

    def load_csv_to_sqlite(self, csv_file_path):
        """Load data from CSV file into SQLite database"""
        try:
            # Create table if it doesn't exist
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    ProductID INTEGER PRIMARY KEY,
                    ProductName TEXT,
                    Category TEXT,
                    Price REAL,
                    Rating REAL,
                    ReviewCount INTEGER,
                    Stock INTEGER,
                    Discount TEXT,
                    Brand TEXT,
                    LaunchDate TEXT
                )
            """)
            
            # Clear existing data
            self.cursor.execute("DELETE FROM products")
            
            # Read CSV and insert into SQLite
            with open(csv_file_path, 'r') as file:
                csv_reader = csv.DictReader(file)
                
                for row in csv_reader:
                    # Clean up the discount percentage
                    discount = row['Discount'].replace('%', '') if '%' in row['Discount'] else row['Discount']
                    
                    # Convert date format if needed (from DD-MM-YYYY to YYYY-MM-DD)
                    try:
                        launch_date = datetime.strptime(row['LaunchDate'], '%d-%m-%Y').strftime('%Y-%m-%d')
                    except ValueError:
                        launch_date = row['LaunchDate']  # fallback to original if parsing fails
                    
                    self.cursor.execute("""
                        INSERT INTO products VALUES (
                            :ProductID, :ProductName, :Category, :Price, :Rating, 
                            :ReviewCount, :Stock, :Discount, :Brand, :LaunchDate
                        )
                    """, {
                        'ProductID': row['ProductID'],
                        'ProductName': row['ProductName'],
                        'Category': row['Category'],
                        'Price': row['Price'],
                        'Rating': row['Rating'],
                        'ReviewCount': row['ReviewCount'],
                        'Stock': row['Stock'],
                        'Discount': discount,
                        'Brand': row['Brand'],
                        'LaunchDate': launch_date
                    })
            
            self.conn.commit()
            print(f"Loaded data from {csv_file_path} into SQLite successfully")
            return True
                
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_file_path}")
            return False
        except Exception as e:
            print(f"Error loading CSV to SQLite: {e}")
            return False

    def generate_sql_query(self, user_input):
        """Generate SQL query using DeepSeek model"""
        try:
            prompt = f"""Convert this natural language query to SQL for SQLite.
            Database table 'products' has columns: ProductID, ProductName, Category, Price, Rating, 
            ReviewCount, Stock, Discount, Brand, LaunchDate.
            
            Important Rules:
            1. When asking for "highest" or "most expensive", use ORDER BY with DESC and LIMIT 1
            2. When asking for a specific column, only select that column
            3. Always respond with just the SQL query, nothing else
            
            Examples:
            Input: Which product has the highest price?
            Output: SELECT * FROM products ORDER BY Price DESC LIMIT 1
            
            Input: Show me the name of the most expensive product
            Output: SELECT ProductName FROM products ORDER BY Price DESC LIMIT 1
            
            Input: Find products with price > 50
            Output: SELECT * FROM products WHERE Price > 50
            
            Now convert this:
            Input: {user_input}
            Output:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm_model.device)
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract SQL query
            query_match = re.search(r'(SELECT\s+.*?\s+FROM\s+.*?(?:\s+WHERE\s+.*?)?(?:\s+ORDER BY\s+.*?)?(?:\s+LIMIT\s+\d+)?)(?=\n|$)', 
                                  generated_text, re.IGNORECASE)
            if query_match:
                generated_query = query_match.group(1).strip()
                # Post-processing to ensure we get just the product name if that's what was asked
                if "productname" in user_input.lower() and "productname" not in generated_query.lower():
                    generated_query = generated_query.replace("SELECT *", "SELECT ProductName")
                
                self.queries_generated.append(f"User Input: {user_input}\nGenerated Query: {generated_query}\n")
                return generated_query
            
            print("Failed to extract valid SQL query from model response")
            return None
            
        except Exception as e:
            print(f"Error generating SQL query: {e}")
            return None

    def execute_query(self, query):
        """Execute SQL query and return results"""
        try:
            if not query or not query.upper().startswith('SELECT'):
                print("Invalid SQL query - must start with SELECT")
                return None
                
            self.cursor.execute(query)
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
            
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return None

    def display_results(self, results):
        """Display query results"""
        if not results:
            print("No results to display")
            return
            
        print("\nQuery Results:")
        print("-" * 80)
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            for key, value in doc.items():
                print(f"  {key}: {value}")
            print("-" * 80)

    def save_results_to_csv(self, results, output_file):
        """Save results to CSV file"""
        try:
            if not results:
                print("No results to save")
                return False
                
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            
            print(f"Results saved to {output_file}")
            return True
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
            return False

    def save_generated_queries(self, filename="Queries_generated.txt"):
        """Save all generated queries to a file"""
        try:
            with open(filename, 'w') as f:
                f.writelines(self.queries_generated)
            print(f"Generated queries saved to {filename}")
        except Exception as e:
            print(f"Error saving generated queries: {e}")

    def close(self):
        """Clean up resources"""
        if self.conn:
            self.conn.close()
        print("System shutdown complete")

def main():
    print("Starting product query system...")
    try:
        system = DataQuerySystem()
        
        # Load CSV data
        csv_file = input("Enter CSV file path [sample_data.csv]: ").strip() or "sample_data.csv"
        if not system.load_csv_to_sqlite(csv_file):
            return
        
        # Main interaction loop
        while True:
            print("\n1. New query\n2. Exit")
            choice = input("Choose option (1-2): ").strip()
            
            if choice == '2':
                break
            elif choice != '1':
                print("Invalid choice")
                continue
                
            # Get and process query
            print("\nAvailable columns: ProductID, ProductName, Category, Price, Rating, ReviewCount, Stock, Discount, Brand, LaunchDate")
            user_input = input("\nEnter your query: ").strip()
            
            if not user_input:
                print("Query cannot be empty")
                continue
                
            query = system.generate_sql_query(user_input)
            if not query:
                continue
                
            print(f"\nGenerated SQL: {query}")
            results = system.execute_query(query)
            
            if not results:
                continue
                
            # Handle results
            print(f"\nFound {len(results)} results")
            action = input("1. Display\n2. Save CSV\n3. Both\nChoose (1-3): ").strip()
            
            if action in ['1', '3']:
                system.display_results(results)
            if action in ['2', '3']:
                output_file = input("Output filename [results.csv]: ").strip() or "results.csv"
                system.save_results_to_csv(results, output_file)
        
        # Save all generated queries
        system.save_generated_queries()
        system.close()
        
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        print("Program ended")

if __name__ == "__main__":
    main()
