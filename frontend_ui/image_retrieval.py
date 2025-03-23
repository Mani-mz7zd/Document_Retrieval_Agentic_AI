import streamlit as st
import requests
import json
from PIL import Image
from io import BytesIO
from database import init_db, add_user, verify_user, get_textbook_metadata
from streamlit_lottie import st_lottie
from concurrent.futures import ThreadPoolExecutor
from anthropic_client   import AnthropicClient
import os
from pathlib import Path

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", '')
SOURCE_DATA_DIRECTORY = os.getenv("SOURCE_DATA_DIRECTORY", '')
SOURCE_THUMBNAIL_DIRECTORY = os.getenv("SOURCE_THUMBNAIL_DIRECTORY", '')

st.set_page_config(
        page_title="Document Search System",
        page_icon="🔍",
        layout="wide"
    )

init_db()

# Backend API URL
BACKEND_URL = "http://backend:8000"

def load_lottiefile(filepath: str):
    with open(filepath, "r") as file:
        return json.load(file)

lottie_animation = load_lottiefile("Animation - 1731620804494.json")

st.markdown("""
<style>
    .login-container {
        max-width: 400px;
        padding: 20px;
        margin: auto;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    .stTextInput > div > div > input {
        border-radius: 5px;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        margin-top: 10px;
    }
    .main-header {
        text-align: center;
        padding: 20px;
    }
    .search-title {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #1E3D59 !important;
        margin-bottom: 10px !important;
    }
    .section-header {
        font-size: 28px !important;
        font-weight: bold !important;
        color: #1E3D59 !important;
        margin: 30px 0 20px 0 !important;
        padding-bottom: 10px !important;
        border-bottom: 2px solid #1E3D59;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .search-metadata {
        font-size: 16px !important;
        color: #666666 !important;
        margin: 5px 0 !important;
        line-height: 1.5 !important;
    }
</style>
""", unsafe_allow_html=True)

def login_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st_lottie(lottie_animation, height=400, key="login_animation")
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Login", key="login_btn"):
                if verify_user(username, password):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        with col_btn2:
            if st.button("Register", key="register_btn"):
                st.session_state['show_register'] = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def register_page():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st_lottie(lottie_animation, height=400, key="register_animation")
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Register</h2>", unsafe_allow_html=True)
        
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("Register", key="register_submit"):
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif not username or not email or not password:
                    st.error("Please fill all fields")
                else:
                    if add_user(username, password, email):
                        st.success("Registration successful! Please login.")
                        st.session_state['show_register'] = False
                        st.rerun()
                    else:
                        st.error("Username or email already exists")
        
        with col_btn2:
            if st.button("Back to Login"):
                st.session_state['show_register'] = False
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    
    if 'show_register' not in st.session_state:
        st.session_state['show_register'] = False
    
    if not st.session_state['logged_in']:
        st.markdown("<h1 class='main-header'>Document Search and Retrieval System</h1>", unsafe_allow_html=True)
        if st.session_state['show_register']:
            register_page()
        else:
            login_page()
        return

    st.title("Document Search and Retrieval System")
    st.sidebar.title("Navigation")
    
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
    
    st.sidebar.write(f"Welcome, {st.session_state['username']}!")
    
    page = st.sidebar.radio("Choose a page", ["Index Images", "Smart Search", "Qdrant Collections Management"])

    # if page == "Document Upload":
    #     document_upload_page()
    if page == "Smart Search":
        image_search_page()
    elif page == "Qdrant Collections Management":
        qdrant_collections_page()
    elif page == "Index Images":
        document_upload_page()

# Images embed and indexing to Qdrant vector store
def document_upload_page():
    st.header("Document Upload")
    st.write("Upload image file to index in the database")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['pdf', 'zip', 'png', 'jpg', 'jpeg'],
        # type=['png', 'jpg', 'jpeg'],
        # help="Supported formats: PDF, ZIP (containing images), PNG, JPG, JPEG"
        help="Supported formats: ZIP (containing images), PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        if st.button("Process and Index Document"):
            with st.spinner("Processing and indexing document..."):
                try:
                    files = {"file": uploaded_file}
                    response = requests.post(f"{BACKEND_URL}/document_embed", files=files)
                    
                    if response.status_code == 200:
                        st.success("Document successfully processed and indexed!")
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")



def display_search_results(results, query):
    # Group results by ISBN
    grouped_results = {}
    retrieved_points = results.get('retrieved_image_points', [])
    for item in retrieved_points:
        isbn = item['ISBN']
        if isbn not in grouped_results:
            metadata = get_textbook_metadata(isbn)
            grouped_results[isbn] = {
                'metadata': metadata,
                'pages': []
            }
        grouped_results[isbn]['pages'].append({'path': item['image'], 'score': item['score'],'page_number': item['page_number']})

    # Display textbook results
    st.markdown('<h2 class="section-header">Search Results</h2>', unsafe_allow_html=True)
    
    for isbn, data in grouped_results.items():
        metadata = data['metadata']
        metadata['thumbnail_location'] = SOURCE_THUMBNAIL_DIRECTORY + metadata['thumbnail_location'].split('/')[-1]
        
        with st.container():
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if os.path.exists(metadata['thumbnail_location']):
                    st.image(metadata['thumbnail_location'], width=150)

            with col2:
                summary = metadata["summary"] if metadata["summary"] else 'N/A'
                st.markdown(
                    f'<div style="height: 180px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; border-radius: 10px;">'
                    f'<div class="search-title">{metadata["title"]}</div>'
                    f'<div class="search-metadata">'
                    f'<strong>Authors:</strong> {", ".join(metadata["main_authors"] + metadata["related_authors"])}<br>'
                    f'<strong>Published:</strong> {metadata["publisher"]} ({metadata["published_year"]})<br>'
                    f'<strong>Edition:</strong> {metadata["edition"]}<br>'
                    f'<strong>Language(s):</strong> {", ".join(metadata["languages"])}<br>'
                    f'<strong>Subjects:</strong> {", ".join(metadata["subjects"])}<br>'
                    f'<strong>Summary:</strong> {summary}<br>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Display top 3 relevant pages across all books
    if grouped_results:
        st.markdown('<h2 class="section-header">Retrieved Pages</h2>', unsafe_allow_html=True)
        
        # Flatten and sort all pages by score
        all_pages = []
        for isbn, data in grouped_results.items():
            for page in data['pages']:
                all_pages.append({
                    'isbn': isbn,
                    'metadata': data['metadata'],
                    'path': page['path'],
                    'score': page['score'],
                    'page_number': page['page_number']
                })
        
        # Sort by score in descending order and take top 3
        all_pages.sort(key=lambda x: x['score'], reverse=True)
        top_3_pages = all_pages[:3]

        # Display in columns
        cols = st.columns(3)
        for idx, page_data in enumerate(top_3_pages):
            with cols[idx]:
                try:
                    image = Image.open(page_data['path'])
                    st.image(image, use_container_width=True)
                    
                    # Display metadata for the page
                    st.markdown(
                        f'<div class="search-metadata" style="text-align: center;">'
                        f'<strong>From:</strong> {page_data["metadata"]["title"]}<br>'
                        f'<strong>Page:</strong> {page_data["page_number"]}<br>'
                        f'<strong>Relevance Score:</strong> {page_data["score"]:.2f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")


# Main Image search Page
def image_search_page():
    st.header("Smart Search")
    st.write("AI-powered image search for text queries and Answer generation")
    
    query = st.text_input("Enter your search query")
    
    if st.button("Search"):
        if query:
            with st.spinner("Searching..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/document_retrieval",
                        json={"user_query": query}
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        if results:
                            for item in results.get('retrieved_image_points', []):
                                if not item['image'].startswith("/home"):
                                    item['image'] = SOURCE_DATA_DIRECTORY + item['image'].split('/')[-1]
                                else:
                                    item['image'] = SOURCE_DATA_DIRECTORY + item['image'].split('/')[-1]

                            # Generate and display AI response
                            with st.container():
                                st.markdown('<h2 class="section-header">AI Generated Response</h2>', unsafe_allow_html=True)
                                with st.spinner("Generating answer..."):
                                    try:
                                        # Initialize Anthropic client
                                        client = AnthropicClient(api_key=ANTHROPIC_API_KEY)

                                        # Prepare the prompt
                                        prompt = f"""You are an expert in interpreting and understanding the content in the images. Using the image as a reference, answer then question. Be very straight to the point and do not include additional information. Here is the question: {query}"""

                                        def get_llm_response():
                                            return client.send_message(
                                                content=prompt,
                                                image_paths=results.get('retrieved_image_points', []),
                                                max_tokens=1000,
                                                temperature=0
                                            )

                                        # Use ThreadPoolExecutor for non-blocking execution
                                        with ThreadPoolExecutor() as executor:
                                            future = executor.submit(get_llm_response)

                                            # Get the response
                                            response = future.result()

                                            if response['status']:
                                                st.markdown(
                                                    f'<div class="search-metadata" style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; height: 200px; overflow-y: auto;">'
                                                    f'{response["result"]}'
                                                    f'</div>',
                                                    unsafe_allow_html=True)
                                            else:
                                                st.markdown(
                                                    f'<div class="search-metadata" style="color: #FF0000;">'
                                                    f'Failed to generate response: {response.get("error", "Unknown error")}'
                                                    f'</div>',
                                                    unsafe_allow_html=True
                                                )
                                    except Exception as e:
                                        st.markdown(
                                            f'<div class="search-metadata" style="color: #FF0000;">'
                                            f'An error occurred: {str(e)}'
                                            f'</div>',
                                            unsafe_allow_html=True
                                        )
                            
                            # Display search results
                            display_search_results(results, query)
                            
                        else:
                            st.warning("No results found.")
                    else:
                        st.error("Error in search request")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a search query")

def qdrant_collections_page():
    st.header("Qdrant Collections Management")
    
    st.subheader("Create New Collection")
    col1, col2, col3 = st.columns(3)
    with col1:
        collection_name = st.text_input("Collection Name")
    with col2:
        vector_size = st.number_input("Vector Size", min_value=1, value=768)
    with col3:
        indexing_threshold = st.number_input("Indexing Threshold", min_value=1, value=20000)
    
    if st.button("Create Collection"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/create_qdrant_collection",
                json={
                    "collection_name": collection_name,
                    "vector_size": vector_size,
                    "indexing_threshold": indexing_threshold
                }
            )
            if response.status_code == 200:
                st.success("Collection created successfully!")
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.subheader("Existing Collections")
    if st.button("Refresh Collections List"):
        try:
            response = requests.post(f"{BACKEND_URL}/get_qdrant_collections")
            if response.status_code == 200:
                collections = response.json()
                print(collections)
                if collections:
                    for collection in collections['collections']:
                        st.write(f"- {collection['name']}")
                else:
                    st.info("No collections found")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # st.subheader("Delete Collection")
    # collection_to_delete = st.text_input("Enter collection name to delete")
    # if st.button("Delete Collection"):
    #     if collection_to_delete:
    #         try:
    #             response = requests.post(
    #                 f"{BACKEND_URL}/delete_qdrant_collection",
    #                 json={"collection_name": collection_to_delete}
    #             )
    #             if response.status_code == 200:
    #                 st.success(f"Collection '{collection_to_delete}' deleted successfully!")
    #             else:
    #                 st.error(f"Error: {response.json()['detail']}")
    #         except Exception as e:
    #             st.error(f"An error occurred: {str(e)}")
    #     else:
    #         st.warning("Please enter a collection name to delete")

if __name__ == "__main__":
    main()