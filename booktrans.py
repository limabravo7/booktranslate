#v0.9: Flags: language settings (--from-lang DE --to-lang EN), processing --mode (--batch, --resume, --test), model selection (--model gpt-4o-mini), save temp files (--debug). Can resume expired or cancelled batch jobs. All jobs can be resumed in either fast or batch mode. Custom prompt can be loaded from customprompt.txt. PDF jobs can produce facing bilingual output.

import argparse
import re
import yaml
import os
from openai import OpenAI
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import signal
import time
from epub_handler import EPUBHandler
from html_processor import split_html_by_paragraph
from pdf_handler import PDFHandler

# Add UTC constant
UTC = timezone.utc
DEBUG = False  # Global debug flag

class BatchProcessingError(Exception):
    """Custom exception for batch processing errors"""
    pass

def ensure_dir(name):
    """Create directory relative to script location"""
    dir_path = Path(__file__).parent / name
    dir_path.mkdir(exist_ok=True)
    return dir_path

def chmod_recursive(path):
    """Recursively change permissions"""
    try:
        os.chmod(path, 0o777)  # Full permissions for owner, group, others
        
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for d in dirs:
                    os.chmod(os.path.join(root, d), 0o777)
                for f in files:
                    os.chmod(os.path.join(root, f), 0o777)
    except Exception as e:
        print(f"Warning: Failed to change permissions for {path}: {e} [chmod_recursive]")

def cleanup_files(client, file_ids, temp_dir=None, keep_temp=False):
    """Clean up temporary files and directories with error handling.
    temp_dir: Path to the specific job directory to clean up
    file_ids: List of OpenAI file IDs to delete
    keep_temp: If True, preserve the job directory
    """
    # First print the keep_temp status
    print(f"Cleanup called with keep_temp={keep_temp} for job dir: {temp_dir.name} [cleanup_files]")
    
    # Exit early if we should keep temp files
    if keep_temp:
        print(f"Keeping job directory: {temp_dir} [cleanup_files]")
        return

    if not keep_temp and temp_dir and temp_dir.exists():
        try:
            print(f"\nStarting cleanup of job directory: {temp_dir.name} [cleanup_files]")
            
            # Force permissions on the directory and its contents
            chmod_recursive(temp_dir)
            
            # Remove all files in the job directory
            for item in temp_dir.glob('*'):
                if item.is_file():
                    try:
                        # Try to force close any open handles to the file
                        with open(item, 'r') as f:
                            try:
                                os.fsync(f.fileno())
                            except:
                                pass
                        
                        item.unlink(missing_ok=True)
                        #print(f"Removed file: {item} [cleanup_files]")
                    except Exception as e:
                        print(f"Warning: Could not remove file {item}: {e} [cleanup_files]")

            # Finally try to remove the job directory itself
            try:
                temp_dir.rmdir()
                print(f"Removed job directory: {temp_dir.name} [cleanup_files]")
            except Exception as e:
                print(f"Warning: Could not remove job directory {temp_dir.name}: {e} [cleanup_files]")
                
        except Exception as e:
            print(f"Warning: Error during cleanup: {e} [cleanup_files]")
            
    # Clean up OpenAI files
    for file_id in file_ids:
        try:
            client.files.delete(file_id)
            print(f"Deleted OpenAI file: {file_id} [cleanup_files]")
        except Exception as e:
            print(f"Warning: Failed to delete OpenAI file {file_id}: {e} [cleanup_files]")

def read_config():
    config_file = 'config.yaml'
    try:
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except FileNotFoundError:
        print(f"Error: {config_file} not found. Please create it with your OpenAI API key. [read_config]")
        print("Example config.yaml content:\nopenai:\n  api_key: 'your-api-key-here' [read_config]")
        exit(1)

def create_job_id(input_epub_path, from_lang, to_lang, model, timestamp=None):
    """Create unique job ID based on input parameters and timestamp"""
    if timestamp is None:
        timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
    base_name = Path(input_epub_path).stem
    return f"{base_name}_{from_lang}_{to_lang}_{model}_{timestamp}"

def ensure_temp_structure(job_id):
    """Create and return paths for all temporary files"""
    temp_dir = ensure_dir("temp")
    job_dir = temp_dir / job_id
    job_dir.mkdir(exist_ok=True)
    
    paths = {
        'job_dir': job_dir,
        'state_file': job_dir / 'job_state.json',
        'chunks_file': job_dir / 'chunks.json',  # Store all original chunks
        'translations_file': job_dir / 'translations.json',  # Store completed translations
        'progress_log': job_dir / 'progress.log'  # Log translation progress
    }
    return paths

def save_job_state(paths, chunks_total, chunks_completed, translations):
    """Save current job state and translations"""
    state = {
        'chunks_total': chunks_total,
        'chunks_completed': chunks_completed,
        'last_updated': datetime.now(UTC).isoformat()
    }
    
    try:
        # Save state with fsync for durability
        with open(paths['state_file'], 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # Save translations separately with fsync
        with open(paths['translations_file'], 'w', encoding='utf-8') as f:
            json.dump(translations, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
            
    except Exception as e:
        print(f"Warning: Could not save state: {e} [save_job_state]")
        # Continue processing but warn user
        print("Warning: Progress may not be resumable if the script is interrupted [save_job_state]")

def load_job_state(paths):
    """Load existing state and translations, making job_state.json optional"""
    state = {}
    try:
        # Load chunks.json (required)
        print(f"Loading chunks from: {paths['chunks_file']} [load_job_state]")
        if not paths['chunks_file'].exists():
            print(f"No chunks file found at: {paths['chunks_file']} [load_job_state]")
            return None
            
        with open(paths['chunks_file'], 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            # Explicitly reconstruct chunks as list of tuples (id, text)
            state['chunks'] = [(id, text) for id, text in chunks_data['chunks']]
            # Properly reconstruct chapter_map from the stored format
            state['chapter_map'] = {
                chunk_id: (data['item'], data['pos']) 
                for chunk_id, data in chunks_data['chapter_map'].items()
            }

        # Add total chunks count
        state['chunks_total'] = len(state['chunks'])
            
        # Load translations.json (optional)
        print(f"Loading translations from: {paths['translations_file']} [load_job_state]")
        if paths['translations_file'].exists():
            with open(paths['translations_file'], 'r', encoding='utf-8') as f:
                state['translations'] = json.load(f)
                state['chunks_completed'] = len(state['translations'])
        else:
            state['translations'] = {}
            state['chunks_completed'] = 0

        # Extract timestamp from job_id directory name if exists
        if paths['job_dir'].exists():
            timestamp = paths['job_dir'].name.split('_')[-1]
            state['last_updated'] = f"{timestamp[:8]}T{timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}.000000+00:00"
                
    except Exception as e:
        print(f"Warning: Could not load state files: {e} [load_job_state]")
        print(f"Attempted to load from paths: [load_job_state]")
        for key, path in paths.items():
            print(f"  {key}: {path} (exists: {path.exists()}) [load_job_state]")
        return None
        
    return state

def save_chunks(paths, all_chunks, chapter_map):
    """Save initial chunks and chapter mapping"""
    chunks_data = {
        'chunks': [(id, text) for id, text in all_chunks],
        'chapter_map': {id: {'item': str(item), 'pos': pos} for id, (item, pos) in chapter_map.items()}
    }
    with open(paths['chunks_file'], 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, indent=2)

def log_progress(paths, message):
    """Log progress with timestamp"""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    try:
        with open(paths['progress_log'], 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        print(f"Warning: Could not write to progress log: {e} [log_progress]")
        print(f"Progress: {message} [log_progress]")

def system_prompt(from_lang, to_lang, filetype):
    custom_prompt_path = Path("./customprompt.txt")
    custom_epubprompt_path = Path("./customepubprompt.txt")
    custom_pdfprompt_path = Path("./custompdfprompt.txt")
    if (custom_prompt_path.exists()):
        print(f"Using customprompt.txt [system_prompt]")
        with open(custom_prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    if filetype == 'epub':
        if (custom_epubprompt_path.exists()):
            print(f"Using customepubprompt.txt [system_prompt]")
            with open(custom_epubprompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"""You are an academic translator, translating to {to_lang}.
CRITICAL: You must preserve ALL HTML/XML structure exactly as provided.
You don't need to translate quotations in Greek or Latin.
If you find numbers which look like footnote markers, and make them superscript.
"""
    elif filetype == 'pdf':
        if (custom_pdfprompt_path.exists()):
            print(f"Using custompdfprompt.txt [system_prompt]")
            with open(custom_pdfprompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            return f"""You are an academic translator, translating to {to_lang}.
CRITICAL: You must preserve ALL HTML/XML structure exactly as provided.
You don't need to translate quotations in Greek or Latin.
"""

def load_test_translations(test_file):
    """Load test translations from a JSON file with flat chunk IDs"""
    test_path = Path("test") / Path(test_file).stem / "translations.json"
    
    if not test_path.exists():
        print(f"Error: Test translations file not found: {test_path} [load_test_translations]")
        return None
    
    try:
        with open(test_path, 'r', encoding='utf-8') as f:
            normalized_test_translations = json.load(f)
        
        if not isinstance(normalized_test_translations, dict):
            print("Warning: Test translations should be a dictionary of chunk_id to translation. [load_test_translations]")
            return {}
        
        print(f"Loaded {len(normalized_test_translations)} test translations [load_test_translations]")
        return normalized_test_translations
    except Exception as e:
        print(f"Warning: Could not load test translations from {test_path}: {e} [load_test_translations]")
        return {}

def save_batch_state(temp_dir, batch_id, input_file_id, timestamp, job_metadata, paths):
    """Save batch processing state for later checking"""
    state_file = temp_dir / f"batch_status_{timestamp}.json"
    state = {
        "batch_id": batch_id,
        "input_file_id": input_file_id,
        "timestamp": timestamp,
        "job_metadata": job_metadata,
        "paths": {k: str(v) for k, v in paths.items()}
    }
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    return state_file

def load_batch_state(temp_dir):
    """Load most recent batch state file"""
    state_files = list(temp_dir.glob("batch_status_*.json"))
    if not state_files:
        return None
    latest_file = max(state_files, key=lambda f: f.stat().st_mtime)
    with open(latest_file, 'r', encoding='utf-8') as f:
        return json.load(f), latest_file

def batch_translate_chunks(client, chunks, from_lang, to_lang, mode=None, model='gpt-4o-mini', 
                         test_translations=None, keep_temp=False, paths=None, chapter_map=None, filetype='epub'):
    """Handle translation of all chunks in a single batch"""
    if mode == 'batchcheck':
        return {}, None, None

    # Extract input_epub_path from the job directory name
    if paths:
        input_epub_path = paths['job_dir'].parent.parent / paths['job_dir'].name.split('_')[0]
    else:
        input_epub_path = "unknown_input"  # Fallback value if paths not provided

    temp_dir = ensure_dir("temp")
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    import select
    import sys
    
    # Create batch input file
    batch_file_path = temp_dir / f"batch_input_{timestamp}.jsonl"
    with open(batch_file_path, "w", encoding="utf-8") as f:
        for chunk_id, chunk_text in chunks:
            request = {
                "custom_id": chunk_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt(from_lang, to_lang, filetype)
                        },
                        {
                            "role": "user",
                            "content": chunk_text
                        }
                    ],
                    "temperature": 0.2
                }
            }
            f.write(json.dumps(request) + "\n")
        
    # Process batch
    print("Uploading batch file... [batch_translate_chunks]")
    batch_file = client.files.create(
        file=open(batch_file_path, "rb"),
        purpose="batch"
    )
    input_file_id = batch_file.id
    
    # Delete batch input file after uploading, unless debug flag is set
    if not DEBUG:
        try:
            batch_file_path.unlink()
            print(f"Deleted batch input file: {batch_file_path} [batch_translate_chunks]")
        except Exception as e:
            print(f"Warning: Could not delete batch input file {batch_file_path}: {e} [batch_translate_chunks]")
    
    print(f"Creating batch job with file ID: {input_file_id} [batch_translate_chunks]")
    batch_job = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    batch_id = batch_job.id
    print(f"Batch job created with ID: {batch_id} [batch_translate_chunks]")
        
    print("\nBatch processing started. Checking initial status in 5 seconds... [batch_translate_chunks]")
    time.sleep(5)
    
    status = client.batches.retrieve(batch_id)
    print(f"Status: {status.status} - Completed: {status.request_counts.completed}/{status.request_counts.total} [batch_translate_chunks]")
    
    if status.status == "failed":
        print("Batch processing failed. [batch_translate_chunks]")
        return {}, input_file_id, status
        
    # Save state and exit if not failed
    job_metadata = {
        "input_file": str(input_epub_path),
        "from_lang": from_lang,
        "to_lang": to_lang,
        "model": model,
        "chapter_map": {chunk_id: {"item": str(item), "pos": pos} 
                       for chunk_id, (item, pos) in chapter_map.items()} if chapter_map else {}
    }
    
    state_file = save_batch_state(temp_dir, batch_id, input_file_id, timestamp, job_metadata, paths)
    print(f"\nBatch processing in progress. Saved state to: {state_file} [batch_translate_chunks]")
    print("Run script again with --mode batchcheck to check status and retrieve results [batch_translate_chunks]")
    return {}, input_file_id, status

def parse_batch_response(response):
    """Parse batch response into translations dictionary"""
    translations = {}
    for line in response.splitlines():
        if not line.strip():
            continue
        result = json.loads(line)
        chunk_id = result['custom_id']
        if 'response' in result and 'body' in result['response']:
            if 'choices' in result['response']['body'] and result['response']['body']['choices']:
                translated_text = result['response']['body']['choices'][0]['message']['content']
                # Remove markdown code block markers if present
                translated_text = re.sub(r'^```[a-zA-Z]*\n', '', translated_text)
                translated_text = re.sub(r'```$', '', translated_text)
                translations[chunk_id] = translated_text
    return translations

def translate_chunk(client, text, chunk_id=None, from_lang='EN', to_lang='PL', model='gpt-4o-mini', test_translations=None, filetype='epub'):
    """Translate a single chunk, with optional test mode"""
    if test_translations is not None:
        # Directly look up translation by chunk_id
        translation = test_translations.get(chunk_id)
        if translation:
            return translation
        else:
            print(f"Warning: No test translation found for chunk {chunk_id} [translate_chunk]")
            return f"[Translation content for chunk-{chunk_id} would go here]"
    
    # Normal API mode
    system_message = system_prompt(from_lang, to_lang, filetype)
    user_message = text
    
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            { 'role': 'system', 'content': system_message },
            { 'role': 'user', 'content': user_message },
        ]
    )
    
    translated_text = response.choices[0].message.content
    # Remove markdown code block markers if present
    translated_text = re.sub(r'^```[a-zA-Z]*\n', '', translated_text)
    translated_text = re.sub(r'```$', '', translated_text)
    return translated_text

def handle_interrupt(signum, frame):
    """Handle keyboard interrupt"""
    print("\nInterrupt received. Saving state and exiting... [handle_interrupt]")
    raise KeyboardInterrupt

def save_translations(paths, translations):
    """Save translations to translations.json with fsync for durability."""
    # Ensure all keys in translations are strings
    translations = {str(k): v for k, v in translations.items()}
    with open(paths['translations_file'], 'w', encoding='utf-8') as f:
        json.dump(translations, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

def process_translations(client, all_chunks, translations, mode, from_lang, to_lang, 
                         paths, model='gpt-4o-mini', test_translations=None, debug=False,
                         chapter_map=None, filetype='epub'):
    """Handle translation based on the specified mode and update translations."""
    if mode == 'batchcheck':
        # Load saved batch state and check status
        temp_dir = ensure_dir("temp")
        state, state_file = load_batch_state(temp_dir)
        if not state:
            print("No batch state found. Run with --mode batch first. [process_translations]")
            return {}, None, None
        
        batch_id = state['batch_id']
        print(f"Checking status for batch {batch_id} [process_translations]")
        status = client.batches.retrieve(batch_id)
        
        if status.status != "completed":
            print(f"Batch status: {status.status} [process_translations]")
            print(f"Progress: {status.request_counts.completed}/{status.request_counts.total} [process_translations]")
            return {}, state['input_file_id'], status
            
        print("Batch completed! Retrieving results... [process_translations]")
        
        # Load existing translations first
        paths = {k: Path(v) for k, v in state['paths'].items()}
        translations = {}
        if Path(paths['translations_file']).exists():
            print("Loading existing translations... [process_translations]")
            with open(paths['translations_file'], 'r', encoding='utf-8') as f:
                translations = json.load(f)
            print(f"Loaded {len(translations)} existing translations [process_translations]")
        
        # Get and merge new translations
        response = client.files.content(status.output_file_id)
        response_text = response.read().decode('utf-8')
        new_translations = parse_batch_response(response_text)
        print(f"Got {len(new_translations)} new translations from batch [process_translations]")
        
        # Merge and save
        translations.update(new_translations)
        save_translations(paths, translations)
        print(f"Total translations after merge: {len(translations)} [process_translations]")
        
        return translations, state['input_file_id'], status

    # Ensure translations keys are strings
    translations = {str(k): v for k, v in translations.items()}
    
    # Define untranslated_chunks once for all modes that require it
    untranslated_chunks = [(chunk_id, chunk_text) for chunk_id, chunk_text in all_chunks if str(chunk_id) not in translations]
    
    if mode == 'resumebatch':
        print(f"\nResuming batch translation for {len(untranslated_chunks)} remaining chunks [process_translations]")
        if not untranslated_chunks:
            print("No untranslated chunks found - all translations complete! [process_translations]")
            return translations, None, None
            
        print(f"Creating new batch job for remaining chunks... [process_translations]")
        new_translations, input_file_id, status = batch_translate_chunks(
            client, untranslated_chunks, from_lang, to_lang, mode='batch', model=model,
            test_translations=test_translations, keep_temp=debug, paths=paths,
            chapter_map=chapter_map, filetype=filetype
        )
        translations.update(new_translations)
        save_translations(paths, translations)
        print(f"Batch job created for remaining chunks. Use --mode batchcheck to monitor progress. [process_translations]")
        return translations, input_file_id, status
        
    if mode in ['fast', 'resume']:
        untranslated = len(untranslated_chunks)
        skipped = len(all_chunks) - untranslated
        
        if skipped > 0:
            print(f"Skipping already translated {skipped} chunks [process_translations]")
        
        print(f"Translating {untranslated} chunks in 'fast' mode [process_translations]")
        for idx, (chunk_id, chunk_text) in enumerate(untranslated_chunks, skipped + 1):
            print(f"Translating chunk #{idx}... [process_translations]")
            try:
                translated_text = translate_chunk(
                    client, chunk_text, chunk_id=chunk_id, from_lang=from_lang, to_lang=to_lang,
                    model=model, test_translations=test_translations, filetype=filetype
                )
                # Ensure chunk_id is a string when saving translation
                translations[str(chunk_id)] = translated_text
                save_translations(paths, translations)
            except Exception as e:
                print(f"Error translating chunk {chunk_id}: {e} [process_translations]")
                raise
        print(f"Total translations after 'fast' mode: {len(translations)} [process_translations]")  # Added logging
        return translations, None, None
    elif mode == 'batch':
        print(f"Translating {len(untranslated_chunks)} chunks in 'batch' mode [process_translations]")
        new_translations, input_file_id, status = batch_translate_chunks(
            client, untranslated_chunks, from_lang, to_lang, mode=mode, model=model,
            test_translations=test_translations, keep_temp=debug, paths=paths,
            chapter_map=chapter_map, filetype=filetype
        )
        translations.update(new_translations)
        save_translations(paths, translations)
        print(f"Total translations after 'batch' mode: {len(translations)} [process_translations]")  # Added logging
        return translations, input_file_id, status
    elif mode == 'test':
        if test_translations:
            for chunk_id, _ in untranslated_chunks:
                if chunk_id in test_translations:
                    translations[str(chunk_id)] = test_translations[chunk_id]
                else:
                    print(f"Warning: No test translation found for chunk {chunk_id} [process_translations]")
                    translations[str(chunk_id)] = f"[TEST MODE] No translation for chunk: {chunk_id}"
        else:
            print("Error: test_translations not provided [process_translations]")
        print(f"Total translations after 'test' mode: {len(translations)} [process_translations]")  # Added logging
        return translations, None, None
    else:
        print(f"Unknown mode: {mode} [process_translations]")
        return translations, None, None

def reassemble_translation(input_epub_path, output_epub_path, chapter_map, translations):
    """Reassemble the translated HTML files and create a new EPUB."""
    epub_handler = EPUBHandler()
    epub_handler.save_translated_epub(input_epub_path, output_epub_path, translations, chapter_map)
    print(f"Processed translation output saved to {output_epub_path} [reassemble_translation]")

def check_batch_status(client, debug=False):
    """Check status of most recent batch job and return state if complete"""
    temp_dir = ensure_dir("temp")
    state_data = load_batch_state(temp_dir)
    
    if not state_data:
        print("No batch state found. Run with --mode batch first. [check_batch_status]")
        return None, None, None
    
    state, state_file = state_data
    
    batch_id = state['batch_id']
    print(f"Checking status for batch {batch_id} [check_batch_status]")
    status = client.batches.retrieve(batch_id)
    
    print(f"Status: {status.status} [check_batch_status]")
    print(f"Progress: {status.request_counts.completed}/{status.request_counts.total} [check_batch_status]")
    
    # Handle expired or cancelled batches with partial results
    if status.status in ['expired', 'cancelled', 'cancelling']:
        if status.request_counts.completed > 0:
            print(f"\nBatch {status.status} but has {status.request_counts.completed} completed requests [check_batch_status]")
            print("Attempting to save partial results... [check_batch_status]")
            
            translations = save_partial_batch_results(client, temp_dir, batch_id, state_file, status)
            if translations:
                print("\nPartial results saved successfully [check_batch_status]")
                print("You can now use --mode resume to complete the remaining translations [check_batch_status]")
        else:
            print(f"\nBatch {status.status} with 0 completed requests [check_batch_status]")
            
        # Only clean up batch state file if debug mode is not set
        if not DEBUG:
            try:
                if state_file.exists():
                    state_file.unlink()
                    print(f"Cleaned up batch state file [check_batch_status]")
            except Exception as e:
                print(f"Warning: Could not remove batch state file: {e} [check_batch_status]")
        else:
            print("Debug mode: Preserving batch state file [check_batch_status]")
            
        return None, None, None
    
    if status.status != "completed":
        return None, None, None
    
    print("Batch completed! Will retrieve results... [check_batch_status]")
    response = client.files.content(status.output_file_id)
    response_text = response.read().decode('utf-8')
    return state, state_file, status

def save_partial_batch_results(client, temp_dir, batch_id, state_file_path, status):
    """Save partial results from an expired/cancelled batch job"""
    try:
        # First download the output file from the API
        if not status.output_file_id:
            print("No output file available [save_partial_batch_results]")
            return None
            
        print(f"Downloading output file {status.output_file_id}... [save_partial_batch_results]")
        response = client.files.content(status.output_file_id)
        output_text = response.read().decode('utf-8')
        
        # Save the raw output file
        output_filename = f"batch_{batch_id}_output.jsonl"
        output_path = temp_dir / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Saved raw output to: {output_path} [save_partial_batch_results]")
            
        translations = {}
        # Process the downloaded content line by line
        for line in output_text.splitlines():
            if not line.strip():
                continue
            try:
                result = json.loads(line)
                chunk_id = result.get('custom_id')
                if not chunk_id:
                    continue
                    
                response = result.get('response', {})
                if not response or response.get('error'):
                    continue
                    
                body = response.get('body', {})
                choices = body.get('choices', [])
                if not choices:
                    continue
                    
                content = choices[0].get('message', {}).get('content')
                if content:
                    translations[chunk_id] = content
            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"Warning: Error processing result for chunk {chunk_id}: {e} [save_partial_batch_results]")
                continue
        
        if not translations:
            print("No valid translations found in partial results [save_partial_batch_results]")
            return None
            
        print(f"Found {len(translations)} valid translations in partial results [save_partial_batch_results]")
        
        # Load the original job state to get the temp directory structure
        with open(state_file_path, 'r', encoding='utf-8') as f:
            original_state = json.load(f)
            
        # Reconstruct the paths dictionary
        paths = {k: Path(v) for k, v in original_state['paths'].items()}
        
        # Save the translations to the original job's translations file
        save_translations(paths, translations)
        print(f"Saved {len(translations)} translations to {paths['translations_file']} [save_partial_batch_results]")
        
        return translations
        
    except Exception as e:
        print(f"Error processing partial results: {e} [save_partial_batch_results]")
        return None

def select_resumable_job(input_epub_path, from_lang, to_lang, model, mode):
    """Helper function to select a resumable job, used by both resume and resumebatch modes"""
    print("\nLooking for resumable translation jobs... [select_resumable_job]")
    resumable_jobs = find_resumable_jobs(input_epub_path, from_lang, to_lang, model)
    if not resumable_jobs:
        print("\nNo resumable jobs found. [select_resumable_job]")
        if mode == 'resumebatch':
            print("Please run with --mode batch first to create a new batch job. [select_resumable_job]")
            sys.exit(1)
        choice = input("Start new translation job? (y/N): [select_resumable_job]")
        if choice.lower() != 'y':
            print("Aborting. [select_resumable_job]")
            return None
        return None
        
    print("\nFound resumable translation jobs: [select_resumable_job]")
    for i, (job_id, timestamp, state) in enumerate(resumable_jobs, 1):
        print(f"{i}. Job from {timestamp} [select_resumable_job]")
        print(f"   Progress: {state['chunks_completed']}/{state['chunks_total']} chunks [select_resumable_job]")
        print(f"   Last updated: {state['last_updated']} [select_resumable_job]")
    
    while True:
        choice = input("\nEnter job number to resume (or 'q' to quit): [select_resumable_job]")
        if choice.lower() == 'q':
            print("Aborting. [select_resumable_job]")
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(resumable_jobs):
                return resumable_jobs[idx][0]
        except ValueError:
            pass
        print("Invalid choice, please try again [select_resumable_job]")

def translate(client, input_path, output_path, from_lang='DE', to_lang='EN', 
              mode=None, model='gpt-4o-mini', fast=False, resume_job_id=None, 
              debug=False, filetype='epub'):
    
    # Initialize success flag at the start
    success = False
    if filetype == 'epub':
        # Early check for batchcheck mode
        if mode == 'batchcheck':
            state, state_file, status = check_batch_status(client, debug)
            if not state:
                return
                
            timestamp = state['timestamp']
            job_id = create_job_id(input_path, from_lang, to_lang, model, timestamp)
            paths = ensure_temp_structure(job_id)
            
            # Let process_translations handle everything including merging existing translations
            translations, input_file_id, status = process_translations(
                client, [], {}, mode, from_lang, to_lang,
                {k: Path(v) for k, v in state['paths'].items()},
                model=model, debug=debug,
                chapter_map=state['job_metadata']['chapter_map'],
                filetype=filetype
            )
            
            if translations:
                # Get chapter_map from the same source that process_translations used
                chapter_map = {
                    chunk_id: (data["item"], data["pos"])
                    for chunk_id, data in state['job_metadata']['chapter_map'].items()
                }
                reassemble_translation(input_path, output_path, chapter_map, translations)
                print(f"Translated EPUB saved to {output_path} [translate]")
                
                # Clean up both job directory and OpenAI files if debug is not set
                if not DEBUG:
                    print(f"Cleaning up job directory and batch files... [translate]")
                    file_ids = []
                    if input_file_id:
                        file_ids.append(input_file_id)
                    if status and status.output_file_id:
                        file_ids.append(status.output_file_id)
                    # Use paths from the state to clean up correct job directory
                    job_dir = Path(state['paths']['job_dir'])
                    cleanup_files(client, file_ids, temp_dir=job_dir, keep_temp=False)
                    
                    # Also clean up the batch state file
                    try:
                        if state_file.exists():
                            state_file.unlink()
                            print(f"Cleaned up batch state file [translate]")
                    except Exception as e:
                        print(f"Warning: Could not remove batch state file: {e} [translate]")
                else:
                    print("Debug mode: Preserving temporary files [translate]")
                
            return

        # Set up interrupt handler at start of function
        original_handler = signal.signal(signal.SIGINT, handle_interrupt)
        interrupted = False
        
        test_translations = None
        if mode == 'test':
            test_translations = load_test_translations(input_path)
            if test_translations is None:
                return

        input_file_id = None
        status = None
        
        try:
            timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
            
            # Handle both resume and resumebatch modes job selection
            # Only call select_resumable_job if we didn't get a job_id from main
            if mode in ['resume', 'resumebatch'] and resume_job_id is None:
                resume_job_id = select_resumable_job(input_path, from_lang, to_lang, model, mode)
                if resume_job_id is None:
                    print("User chose to quit. [translate]")
                    return
                
            if resume_job_id:
                job_id = resume_job_id
                print(f"Resuming job: {job_id} [translate]")
                paths = ensure_temp_structure(job_id)
                
                # Try to load existing state
                existing_state = load_job_state(paths)
                if not existing_state:
                    print(f"No valid state found in {job_id}, starting fresh translation [translate]")
                    resume_job_id = None
                    job_id = create_job_id(input_path, from_lang, to_lang, model)
                    paths = ensure_temp_structure(job_id)
                    existing_state = None
            else:
                job_id = create_job_id(input_path, from_lang, to_lang, model)
                print(f"Starting new job: {job_id} [translate]")
                paths = ensure_temp_structure(job_id)
                existing_state = None

            log_progress(paths, f"{'Resuming' if resume_job_id else 'Starting'} translation job: {job_id}")
            
            success = False
            all_chunks = []
            chapter_map = {}
            translations = {}
            
            try:
                print(f"Started at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC [translate]")
                print(f"Using model: {model} [translate]")
                
                # Use existing state if resuming
                if resume_job_id:
                    print("\nResuming from previous state: [translate]")
                    print(f"Total chunks: {existing_state['chunks_total']} [translate]")
                    print(f"Completed translations: {len(existing_state.get('translations', {}))} [translate]")
                    
                    # Use existing data
                    all_chunks = existing_state['chunks']
                    chapter_map = existing_state['chapter_map']
                    translations = existing_state.get('translations', {})
                    
                    print(f"Resumed with {len(translations)} existing translations [translate]")
                    
                else:
                    # Process EPUB only if not resuming
                    all_chunks, chapter_map = EPUBHandler.build_chunks(input_path)
                    save_chunks(paths, all_chunks, chapter_map)
                    translations = {}

                print(f"Total chunks to process: {len(all_chunks)} [translate]")

                # Determine mode based on flags - IMPORTANT: preserve mode if resuming
                if not mode:
                    mode = 'fast' if fast else 'batch'
                    
                # Process translations using the helper function
                translations, input_file_id, status = process_translations(
                    client, all_chunks, translations, mode, from_lang, to_lang,
                    paths, model=model, test_translations=test_translations,
                    debug=debug, chapter_map=chapter_map, filetype=filetype
                )
                
                print(f"Final translation count: {len(translations)} [translate]")
                
                # Save final state before reassembly
                save_translations(paths, translations)
                
                # Add early return for empty translations
                if len(translations) == 0:
                    print("No translations available. Exiting without creating output file. [translate]")
                    return
                
                # Only reassemble if we have translations
                reassemble_translation(input_path, output_path, chapter_map, translations)
                
                print(f"Translated EPUB saved to {output_path} [translate]")
                success = True

            except KeyboardInterrupt:
                interrupted = True
                print("\nInterrupted by user. Progress is saved and can be resumed with --mode resume [translate]")
                return
                    
            except Exception as e:
                print(f"Error during processing: {e} [translate]")
                raise

        finally:
            signal.signal(signal.SIGINT, original_handler)
            
            # Skip cleanup if interrupted
            if interrupted:
                return
            
            # Determine keep_temp based on debug flags and success state
            keep_temp = DEBUG or not success
            
            # Perform cleanup if necessary
            if not DEBUG:
                print(f"Cleaning up job directory: {paths['job_dir']} [translate]")
                # Only include OpenAI file IDs if they exist
                file_ids = []
                if input_file_id:
                    file_ids.append(input_file_id)
                if status and status.output_file_id:
                    file_ids.append(status.output_file_id)
                cleanup_files(client, file_ids, temp_dir=paths['job_dir'], keep_temp=keep_temp)
            else:
                print(f"Preserving temporary files (keep_temp={keep_temp}) [translate]")
            
            if success:
                print("\nProcessing completed successfully [translate]")
                if DEBUG:
                    print("Debug mode: Temporary files preserved in 'temp' directory [translate]")
            else:
                if mode in ['batchcheck', 'batch'] and status:
                    print(f"\nCurrent batch status: {status.status} [translate]")
                    if status.status == 'in_progress':
                        print("Batch processing is still in progress [translate]")
                        print("Run again with --mode batchcheck to monitor progress [translate]")
                    else:
                        print("\nProcessing failed - temporary files preserved in 'temp' directory [translate]")
                else:
                    print("\nProcessing failed - temporary files preserved in 'temp' directory [translate]")
    elif filetype == 'pdf':
        #Initialise
        success = False
        all_chunks = []
        chapter_map = {}
        translations = {}
        temp_dir = ensure_dir("temp")

        timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S')
        job_id = create_job_id(input_path, from_lang, to_lang, model)
        paths = ensure_temp_structure(job_id)
        print(f"Starting new job: {job_id} [translate]")
        
        print("PDF detected, transcribing... [translate]")

        if mode == 'batchcheck':
            # state_data = load_batch_state(temp_dir)
    
            # if not state_data:
            #     print("No batch state found. Run with --mode batch first. [translate > pdf]")
            #     return
            
            # state, state_file = state_data
            
            # batch_id = state['batch_id']
            # print(f"Checking status for batch {batch_id} [translate > pdf]")
            # status = client.batches.retrieve(batch_id)
            
            # print(f"Status: {status.status} [translate > pdf]")
            # print(f"Progress: {status.request_counts.completed}/{status.request_counts.total} [translate > pdf]")
            
            # if status.status == "completed":
            #     print("Batch completed! Will retrieve results... [translate > pdf]")
            #     response = client.files.content(status.output_file_id)
            #     response_text = response.read().decode('utf-8')
            #     translations = parse_batch_response(response_text)
            #     print(f"Got {len(translations)} translation pages from batch [translate > pdf]")
            return
        # elif mode == 'batch':
        #     print("batch pdf")
        #     batch_file_path = PDFHandler.transcribe_pdf(input_path, paths, dpi=150, batch=True)
            
        #     print("Uploading batch file... [translate > pdf]")
        #     batch_file = client.files.create(
        #             file=open(batch_file_path, "rb"),
        #             purpose="batch"
        #         )
        #     input_file_id = batch_file.id

        #     # Delete batch input file after uploading, unless debug flag is set
        #     if not DEBUG:
        #         try:
        #             batch_file_path.unlink()
        #             print(f"Deleted batch input file: {batch_file_path} [translate > pdf]")
        #         except Exception as e:
        #             print(f"Warning: Could not delete batch input file {batch_file_path}: {e} [translate > pdf]")
    
        #     print(f"Creating batch job with file ID: {input_file_id} [translate > pdf]")
        #     batch_job = client.batches.create(
        #         input_file_id=input_file_id,
        #         endpoint="/v1/chat/completions",
        #         completion_window="24h"
        #     )
            
        #     batch_id = batch_job.id
        #     print(f"Batch job created with ID: {batch_id} [translate > pdf]")
                
        #     print("\nBatch processing started. Checking initial status in 5 seconds... [translate > pdf]")
        #     time.sleep(5)
            
        #     status = client.batches.retrieve(batch_id)
        #     print(f"Status: {status.status} - Completed: {status.request_counts.completed}/{status.request_counts.total} [translate > pdf]")
            
        #     if status.status == "failed":
        #         print("Batch processing failed. [translate > pdf]")
        #         return {}, input_file_id, status
                
        #     # Save state and exit if not failed
        #     job_metadata = {
        #         "input_file": str(input_path),
        #         "from_lang": from_lang,
        #         "to_lang": to_lang,
        #         "model": model,
        #         "chapter_map": {chunk_id: {"item": str(item), "pos": pos} 
        #                     for chunk_id, (item, pos) in chapter_map.items()} if chapter_map else {}
        #     }

        #     state_file = save_batch_state(temp_dir, batch_id, input_file_id, timestamp, job_metadata, paths)
        #     print(f"\nBatch processing in progress. Saved state to: {state_file} [translate > pdf]")
        #     print("Run script again with --mode batchcheck to check status and retrieve results [translate > pdf]")
            

        #     return

        else:
            # Transcribe PDF to HTML
            all_chunks, chapter_map = PDFHandler.transcribe_pdf(client, input_path, paths, dpi=150, batch=False)
            save_chunks(paths, all_chunks, chapter_map)
            print(f"Transcription of {len(all_chunks)} pages complete... [translate]")

            # Translate HTML chunks
            # translations, input_file_id, status = process_translations(
            #             client, all_chunks, translations, mode, from_lang, to_lang,
            #             paths, model=model, test_translations=None,
            #             debug=debug, chapter_map=chapter_map, filetype=filetype
            #         )
            # print(f"Final translation count: {len(translations)} [translate]")

        translations = {chunk_id: chunk for chunk_id, chunk in all_chunks}

        # Save final state before reassembly
        save_translations(paths, translations)
        print(f"Translations saved: {paths['translations_file'].name} [translate]")

        if mode == 'pdfbilingual':
            PDFHandler.save_bilingual_pdf(input_path, translations, output_path)
        else:
            PDFHandler.save_translated_pdf(translations,output_path)

        print(f"Finished at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC [translate]")
        
        success = True

        keep_temp = DEBUG or not success
        if not DEBUG:
            #print(f"Cleaning up job directory: {paths['job_dir']} [translate]")
            # Only include OpenAI file IDs if they exist
            file_ids = []
            cleanup_files(client, file_ids, temp_dir=paths['job_dir'], keep_temp=keep_temp)
        else:
            print(f"Preserving temporary files (keep_temp={keep_temp}) [translate]")

def find_resumable_jobs(input_epub_path, from_lang, to_lang, model):
    """Find all resumable jobs for the given parameters without requiring job_state.json"""
    temp_dir = ensure_dir("temp")
    resumable_jobs = []
    
    # Look for job directories
    prefix = f"{Path(input_epub_path).stem}_{from_lang}_{to_lang}_{model}_"
    for job_dir in temp_dir.iterdir():
        if not job_dir.is_dir():
            continue
            
        try:
            # Check if this is a job directory for our input file
            if not job_dir.name.startswith(prefix):
                print(f"Skipping directory {job_dir.name}: doesn't match pattern {prefix}* [find_resumable_jobs]")
                continue

            paths = ensure_temp_structure(job_dir.name)
            
            # Only require chunks.json and optionally translations.json
            if not paths['chunks_file'].exists():
                print(f"Skipping directory {job_dir.name}: missing chunks.json [find_resumable_jobs]")
                continue
                
            # Count total chunks
            with open(paths['chunks_file'], 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
                chunks_total = len(chunks_data['chunks'])
            
            # Count completed translations
            chunks_completed = 0
            if paths['translations_file'].exists():
                with open(paths['translations_file'], 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                    chunks_completed = len(translations)
            else:
                print(f"Note: Directory {job_dir.name} has no translations.json yet [find_resumable_jobs]")
            
            if chunks_completed >= chunks_total:
                print(f"Found completed job in {job_dir.name}: {chunks_completed}/{chunks_total} chunks [find_resumable_jobs]")
            else:
                print(f"Found resumable job in {job_dir.name}: {chunks_completed}/{chunks_total} chunks completed [find_resumable_jobs]")
            
            # Extract and properly format timestamp from directory name
            date_part = job_dir.name.split('_')[-2]  # Format: YYYYMMDD
            time_part = job_dir.name.split('_')[-1]  # Format: HHMMSS
            
            # Parse each component
            year = date_part[:4]
            month = date_part[4:6]
            day = date_part[6:8]
            hour = time_part[:2]
            minute = time_part[2:4]
            second = time_part[4:6]
            
            raw_timestamp = f"{date_part}_{time_part}"
            padded_timestamp = f"{year}-{month}-{day}T{hour}:{minute}:{second}.000000+00:00"
            
            state = {
                'chunks_total': chunks_total,
                'chunks_completed': chunks_completed,
                'last_updated': padded_timestamp
            }
                
            resumable_jobs.append((job_dir.name, raw_timestamp, state))
                
        except Exception as e:
            print(f"Warning: Could not process directory {job_dir.name}: {e} [find_resumable_jobs]")
            continue
    
    return sorted(resumable_jobs, key=lambda x: x[2]['last_updated'], reverse=True)

def main():
    import sys
    try:
        parser = argparse.ArgumentParser(description='App to translate EPUB and PDF books.')

        parser.add_argument('--input', required=True, help='Input file path.')
        parser.add_argument('--output', help='Output file path. Defaults to [input]_[to-lang]_[model].[ext]')
        parser.add_argument('--from-lang', help='Source language.', default='DE')
        parser.add_argument('--to-lang', help='Target language.', default='EN')
        parser.add_argument('--debug', action='store_true', help='Keep batch files and temp directory for debugging.')
        parser.add_argument('--mode', choices=['batch', 'resume', 'test', 'batchcheck', 'resumebatch', 'pdfbilingual'], 
                           help='Processing mode: "batch" for batch API, '
                                '"resume" to continue previous job, "test" to use test translations, '
                                '"batchcheck" to check batch status, "resumebatch" to resume with batch processing')
        parser.add_argument('--model', help='The model to use for translation (default: gpt-4o-mini)', default='gpt-4o-mini')

        args = parser.parse_args()

        # Set debug mode globally
        global DEBUG
        DEBUG = args.debug

        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            parser.error(f"Input file not found: {args.input} [main]")
        if input_path.suffix.lower() not in ['.epub', '.pdf']:
            parser.error(f"Input file must be an EPUB or PDF file: {args.input} [main]")
        
        # Determine file type
        filetype = input_path.suffix.lower().lstrip('.')

        # Check for resumable jobs if in resume mode
        job_id = None
        if args.mode in ['resume', 'resumebatch']:
            print("\nLooking for resumable translation jobs... [main]")
            resumable_jobs = find_resumable_jobs(args.input, args.from_lang, args.to_lang, args.model)
            if resumable_jobs:
                print("\nFound resumable translation jobs: [main]")
                for i, (job_id, timestamp, state) in enumerate(resumable_jobs, 1):
                    print(f"{i}. Job from {timestamp} [main]")
                    print(f"   Progress: {state['chunks_completed']}/{state['chunks_total']} chunks [main]")
                    print(f"   Last updated: {state['last_updated']} [main]")
                
                while True:
                    choice = input("\nEnter job number to resume (or 'n' for new job, 'q' to quit) [main]:")
                    if choice.lower() == 'n':
                        job_id = None
                        break
                    elif choice.lower() == 'q':
                        print("Aborting. [main]")
                        sys.exit(0)
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(resumable_jobs):
                            job_id = resumable_jobs[idx][0]
                            break
                    except ValueError:
                        pass
                    print("Invalid choice, please try again [main]")
            else:
                print("\nNo resumable jobs found. [main]")
                if args.mode == 'resumebatch':
                    print("Please run with --mode batch first to create a new batch job. [main]")
                    sys.exit(1)
                choice = input("Start new translation job? (y/N): [main]")
                if choice.lower() != 'y':
                    print("Aborting. [main]")
                    sys.exit(0)
                job_id = None

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_dir = ensure_dir("output")
            output_path = str(output_dir / f'{input_path.stem}_{args.to_lang.lower()}_{args.model}.{filetype}')

        config = read_config()
        client = OpenAI(api_key=config['openai']['api_key'])
        
        # Call translate - remove test_translations_file parameter
        translate(client, args.input, output_path, args.from_lang, args.to_lang, 
                 mode=args.mode, model=args.model, 
                 fast=(args.mode not in ['batch']), resume_job_id=job_id, 
                 debug=args.debug, filetype=filetype)
    except KeyboardInterrupt:
        print("\nTranslation interrupted. Use --mode resume to continue later. [main]")
        sys.exit(1)

if __name__ == "__main__":
    main()


