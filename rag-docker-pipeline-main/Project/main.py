import os
import json
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime
import threading
import time

# Import RAG pipeline functions
from rag_pipeline import (
    initialize_settings,
    extract_text,
    create_index,
    query_index,
    clean_text
)

# === Flask App Configuration ===
app = Flask(__name__, 
            static_folder='templates',
            template_folder='templates')
CORS(app)

# Configuration
UPLOAD_FOLDER = './uploads'
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'csv', 'json', 'jpg', 'jpeg', 'png', 'mp3', 'wav', 'mp4', 'zip'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === Global State ===
class RAGState:
    def __init__(self):
        self.index = None
        self.llm_type = None
        self.uploaded_files = []
        self.processing = False
        self.chat_history = []
        self.error_message = None
        self.processing_progress = 0
        self.lock = threading.Lock()

rag_state = RAGState()

# === Helper Functions ===

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_info(file_path):
    """Get file information"""
    try:
        file_size = os.path.getsize(file_path)
        return {
            'name': os.path.basename(file_path),
            'size': file_size,
            'size_formatted': format_file_size(file_size),
            'uploaded_at': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {
            'name': os.path.basename(file_path),
            'size': 0,
            'size_formatted': '0 B',
            'uploaded_at': datetime.now().isoformat()
        }

def format_file_size(size_bytes):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def initialize_llm():
    """Initialize LLM (Groq or Ollama)"""
    try:
        logger.info("🔧 Initializing LLM with Groq...")
        llm_type = initialize_settings(use_groq=True)
        logger.info(f"✅ LLM initialized: {llm_type}")
        return llm_type
    except Exception as e:
        logger.warning(f"⚠️ Groq failed: {e}. Trying Ollama...")
        try:
            llm_type = initialize_settings(use_groq=False)
            logger.info(f"✅ LLM initialized: {llm_type}")
            return llm_type
        except Exception as ollama_error:
            logger.error(f"❌ Both LLMs failed: {ollama_error}")
            rag_state.error_message = "Failed to initialize any LLM (Groq/Ollama)"
            return None

def process_files_background(file_paths):
    """Process files in background thread"""
    try:
        with rag_state.lock:
            rag_state.processing = True
            rag_state.processing_progress = 0
            rag_state.error_message = None
        
        logger.info(f"📂 Processing {len(file_paths)} files...")
        
        # Update progress
        rag_state.processing_progress = 10
        
        # Create index using rag_pipeline
        logger.info("🔨 Creating vector index...")
        index = create_index(file_paths)
        
        # Update progress
        rag_state.processing_progress = 80
        
        if index is None:
            error_msg = "Failed to create index from documents"
            rag_state.error_message = error_msg
            logger.error(f"❌ {error_msg}")
        else:
            with rag_state.lock:
                rag_state.index = index
                rag_state.uploaded_files = [get_file_info(fp) for fp in file_paths]
                rag_state.processing_progress = 100
            logger.info(f"✅ Index created successfully with {len(file_paths)} files")
    
    except Exception as e:
        error_msg = f"Error processing files: {str(e)}"
        logger.error(f"❌ {error_msg}")
        rag_state.error_message = error_msg
    
    finally:
        with rag_state.lock:
            rag_state.processing = False
            time.sleep(0.5)  # Small delay for UI

# === Web Routes ===

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/templates/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('templates', filename)

# === API Routes ===

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'index_ready': rag_state.index is not None,
        'llm_type': rag_state.llm_type,
        'processing': rag_state.processing
    }), 200

@app.route('/api/initialize', methods=['POST'])
def api_initialize():
    """Initialize LLM"""
    try:
        if rag_state.llm_type:
            return jsonify({
                'success': True,
                'message': f'LLM already initialized: {rag_state.llm_type}',
                'llm_type': rag_state.llm_type
            }), 200
        
        llm_type = initialize_llm()
        
        if llm_type is None:
            return jsonify({
                'success': False,
                'error': rag_state.error_message or 'Failed to initialize LLM'
            }), 500
        
        rag_state.llm_type = llm_type
        return jsonify({
            'success': True,
            'message': f'LLM initialized successfully: {llm_type}',
            'llm_type': llm_type
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload and process documents"""
    try:
        # Check if files are provided
        if 'files' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No files provided'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({
                'success': False,
                'error': 'No files selected'
            }), 400
        
        # Initialize LLM if not already done
        if not rag_state.llm_type:
            logger.info("🔧 Auto-initializing LLM...")
            llm_type = initialize_llm()
            if not llm_type:
                return jsonify({
                    'success': False,
                    'error': rag_state.error_message or 'Failed to initialize LLM'
                }), 500
            rag_state.llm_type = llm_type
        
        # Save uploaded files
        saved_files = []
        skipped_files = []
        
        for file in files:
            if file and file.filename:
                if allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    # Add timestamp to avoid filename collisions
                    name, ext = os.path.splitext(filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    unique_filename = f"{name}_{timestamp}{ext}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    
                    file.save(file_path)
                    saved_files.append(file_path)
                    logger.info(f"✅ File uploaded: {filename}")
                else:
                    skipped_files.append(file.filename)
                    logger.warning(f"⚠️ File not allowed: {file.filename}")
        
        if not saved_files:
            return jsonify({
                'success': False,
                'error': 'No valid files were uploaded',
                'skipped_files': skipped_files
            }), 400
        
        # Process files in background thread
        thread = threading.Thread(target=process_files_background, args=(saved_files,))
        thread.daemon = True
        thread.start()
        
        response_data = {
            'success': True,
            'message': f'{len(saved_files)} file(s) uploaded successfully',
            'files': [os.path.basename(f) for f in saved_files],
            'processing': True,
            'llm_type': rag_state.llm_type
        }
        
        if skipped_files:
            response_data['skipped_files'] = skipped_files
            response_data['warning'] = f'{len(skipped_files)} file(s) skipped (unsupported format)'
        
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    """Get processing status"""
    try:
        return jsonify({
            'success': True,
            'processing': rag_state.processing,
            'progress': rag_state.processing_progress,
            'index_ready': rag_state.index is not None,
            'files_count': len(rag_state.uploaded_files),
            'files': rag_state.uploaded_files,
            'llm_type': rag_state.llm_type,
            'error': rag_state.error_message
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/query', methods=['POST'])
def api_query():
    """Query the RAG system"""
    try:
        # Check if index is ready
        if rag_state.index is None:
            return jsonify({
                'success': False,
                'error': 'No documents processed yet. Please upload and process documents first.'
            }), 400
        
        # Check if still processing
        if rag_state.processing:
            return jsonify({
                'success': False,
                'error': 'Documents are still being processed. Please wait...'
            }), 400
        
        # Get query from request
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'Invalid request format'
            }), 400
        
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query cannot be empty'
            }), 400
        
        logger.info(f"🔍 Processing query: {query}")
        
        # Query the index using rag_pipeline
        response = query_index(rag_state.index, query, rag_state.llm_type)

        # Handle None response
        if response is None:
            return jsonify({
                'success': False,
                'error': 'Failed to generate response. Please try again.'
            }), 500

        # Parse response - can be tuple (text, sources) or just text
        sources = []
        if isinstance(response, tuple):
            response_text, sources = response
        else:
            response_text = str(response)
        
        # Add to chat history
        chat_entry = {
            'query': query,
            'response': response_text,
            'sources': sources,
            'timestamp': datetime.now().isoformat(),
            'llm_type': rag_state.llm_type
        }
        
        with rag_state.lock:
            rag_state.chat_history.append(chat_entry)
        
        logger.info(f"✅ Query processed successfully")
        
        return jsonify({
            'success': True,
            'query': query,
            'response': response_text,
            'sources': sources,
            'llm_type': rag_state.llm_type,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        return jsonify({
            'success': False,
            'error': f'Query processing failed: {str(e)}'
        }), 500

@app.route('/api/chat-history', methods=['GET'])
def api_chat_history():
    """Get chat history"""
    try:
        return jsonify({
            'success': True,
            'history': rag_state.chat_history,
            'count': len(rag_state.chat_history)
        }), 200
    
    except Exception as e:
        logger.error(f"❌ History error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    """Clear chat history"""
    try:
        with rag_state.lock:
            rag_state.chat_history = []
        
        logger.info("🗑️ Chat history cleared")
        return jsonify({
            'success': True,
            'message': 'Chat history cleared'
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Clear history error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/reset', methods=['POST'])
def api_reset():
    """Reset entire system"""
    try:
        with rag_state.lock:
            rag_state.index = None
            rag_state.uploaded_files = []
            rag_state.chat_history = []
            rag_state.error_message = None
            rag_state.processing = False
            rag_state.processing_progress = 0
        
        # Clear uploads folder
        try:
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    logger.info(f"🗑️ Deleted: {file}")
        except Exception as e:
            logger.warning(f"⚠️ Error clearing upload folder: {e}")
        
        logger.info("✅ System reset successfully")
        return jsonify({
            'success': True,
            'message': 'System reset successfully'
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Reset error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/files', methods=['GET'])
def api_get_files():
    """Get uploaded files"""
    try:
        return jsonify({
            'success': True,
            'files': rag_state.uploaded_files,
            'count': len(rag_state.uploaded_files)
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Get files error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/info', methods=['GET'])
def api_info():
    """Get system information"""
    try:
        return jsonify({
            'success': True,
            'system_info': {
                'app_name': 'RAG Knowledge Assistant',
                'version': '1.0.0',
                'llm_type': rag_state.llm_type,
                'index_ready': rag_state.index is not None,
                'files_uploaded': len(rag_state.uploaded_files),
                'chat_history_count': len(rag_state.chat_history),
                'processing': rag_state.processing,
                'upload_folder': app.config['UPLOAD_FOLDER'],
                'max_file_size_mb': MAX_FILE_SIZE / (1024*1024),
                'allowed_extensions': list(ALLOWED_EXTENSIONS)
            }
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Info error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/delete-file', methods=['POST'])
def api_delete_file():
    """Delete a specific uploaded file"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({
                'success': False,
                'error': 'Filename not provided'
            }), 400
        
        # Find and remove from uploaded files list
        with rag_state.lock:
            rag_state.uploaded_files = [
                f for f in rag_state.uploaded_files 
                if f['name'] != filename
            ]
        
        # Delete physical file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(file_path):
            os.unlink(file_path)
            logger.info(f"🗑️ Deleted file: {filename}")
        
        return jsonify({
            'success': True,
            'message': f'File {filename} deleted successfully'
        }), 200
    
    except Exception as e:
        logger.error(f"❌ Delete file error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# === Error Handlers ===

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size limit errors"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB'
    }), 413

# === Main Entry Point ===

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 RAG Knowledge Assistant - Starting Server")
    print("="*60)
    logger.info(f"📁 Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    logger.info(f"📊 Max file size: {MAX_FILE_SIZE / (1024*1024):.0f}MB")
    logger.info(f"✅ Allowed file types: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    print("="*60)
    print("🌐 Server running at: http://localhost:5000")
    print("📱 Open your browser and navigate to the URL above")
    print("="*60 + "\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=True,
        threaded=True
    )