import zipfile
from pathlib import Path
from io import BytesIO
from html_processor import split_html_by_paragraph  # Use absolute import

class EPUBHandler:
    """Handler for EPUB file processing"""
    
    @staticmethod
    def extract_content(epub_path):
        """
        Extract content from EPUB file.
        Returns a list of tuples: (filename, content)
        where content is the raw HTML/XHTML content
        """
        content_files = []
        
        with zipfile.ZipFile(epub_path, 'r') as zip_ref:
            # Filter for HTML/XHTML files
            html_files = [f for f in zip_ref.namelist() 
                        if f.endswith(('.html', '.xhtml'))]
            
            for html_file in html_files:
                try:
                    with zip_ref.open(html_file) as file:
                        content = file.read().decode('utf-8')
                        content_files.append((html_file, content))
                except Exception as e:
                    print(f"Warning: Could not process {html_file}: {e}")
                    continue
                    
        return content_files
    
    @staticmethod
    def save_translated_epub(input_epub_path, output_epub_path, translations, chapter_map):
        """
        Create a new EPUB with translated content.
        translations: dict of chunk_id -> translated_text
        chapter_map: dict of chunk_id -> (filename, position)
        """
        
        # Group chunks by filename
        file_chunks = {}
        for chunk_id, (filename, pos) in chapter_map.items():
            if filename not in file_chunks:
                file_chunks[filename] = {}
            if chunk_id in translations:
                file_chunks[filename][pos] = translations[chunk_id]

        # Create new EPUB with translated content
        with zipfile.ZipFile(input_epub_path, 'r') as zip_in:
            with zipfile.ZipFile(output_epub_path, 'w') as zip_out:
                for item in zip_in.infolist():
                    if item.filename in file_chunks:
                        # This is a file that needs translation
                        content = zip_in.read(item.filename).decode('utf-8')
                        
                        chunks = split_html_by_paragraph(content)
                        
                        # Replace chunks with translations
                        for pos, chunk in file_chunks[item.filename].items():
                            if pos < len(chunks):
                                chunks[pos] = chunk
                        
                        # Reassemble the content
                        translated_content = ''.join(chunks)
                        
                        # Create a new ZipInfo to avoid duplicate warnings
                        new_info = zipfile.ZipInfo(filename=item.filename,
                                                 date_time=item.date_time)
                        new_info.compress_type = item.compress_type
                        zip_out.writestr(new_info, translated_content.encode('utf-8'))
                    else:
                        # Copy unchanged files
                        buffer = BytesIO(zip_in.read(item.filename))
                        new_info = zipfile.ZipInfo(filename=item.filename,
                                                 date_time=item.date_time)
                        new_info.compress_type = item.compress_type
                        zip_out.writestr(new_info, buffer.getvalue())
