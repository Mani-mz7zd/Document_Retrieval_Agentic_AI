import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from typing import Dict, Optional
from werkzeug.security import generate_password_hash, check_password_hash

# from dotenv import load_dotenv
# load_dotenv()

# Postgresql Database configuration
DB_CONFIG = {
    'dbname': os.getenv('POSTGRES_DB', 'postgres'),
    'user': os.getenv('POSTGRES_USER', 'postgres'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin'),
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432')
}

def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_user(username, password, email):
    try:
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        password_hash = generate_password_hash(password)
        c.execute('INSERT INTO users (username, password_hash, email) VALUES (?, ?, ?)',
                 (username, password_hash, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verify_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE username = ?', (username,))
    result = c.fetchone()
    conn.close()
    
    if result:
        return check_password_hash(result[0], password)
    return False


def get_db_connection():
    """Create and return a database connection"""
    try:
        conn = psycopg2.connect(
            dbname=DB_CONFIG['dbname'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        raise

def get_textbook_metadata(isbn):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT title, main_authors, publisher, published_year, edition, 
                   related_authors, languages, subjects, summary, thumbnail_location
            FROM textbooks_metadata 
            WHERE isbn = %s
        """, (isbn,))
        result = cur.fetchone()
        if result:
            return {
                'title': result[0],
                'main_authors': result[1],
                'publisher': result[2],
                'published_year': result[3],
                'edition': result[4],
                'related_authors': result[5],
                'languages': result[6],
                'subjects': result[7],
                'summary': result[8],
                'thumbnail_location': result[9]
            }
        return None
    finally:
        cur.close()
        conn.close()