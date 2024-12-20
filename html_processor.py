import re

def split_sentences(text):
    """Split text into sentences while preserving common abbreviations and numerical citations"""
    # Common abbreviations to preserve
    abbreviations = r'(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|vs|e\.g|i\.e|viz|cf|Ch|p|pp|vol|ex|no)\.'
    # Add numerical citations pattern
    numbers = r'\d+'
    # Combine patterns - match either abbreviations or number.number pattern
    no_split_pattern = f'(?:{abbreviations}|{numbers})'
    # Split on period followed by space and capital letter, 
    # but not if preceded by abbreviations or numbers
    sentences = re.split(f'(?<!{no_split_pattern})\\. +(?=[A-Z])', text)
    return [s + '.' for s in sentences[:-1]] + [sentences[-1]]  # Add periods back except for last sentence

def split_html_by_paragraph(html_str, max_chunk_size=10000):
    """Split HTML content by paragraphs, falling back to sentences only when necessary"""
    paragraphs = html_str.split('</p>')
    # Remove empty paragraphs and add closing tags back
    paragraphs = [p.strip() + '</p>' for p in paragraphs if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # If adding this paragraph would exceed max size
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            if current_chunk:
                # Finalize current chunk, start new with this paragraph
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # This paragraph is first in chunk AND too big - split by sentences
                sentences = split_sentences(paragraph)
                temp_chunk = ""
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) > max_chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sentence
                    else:
                        temp_chunk += ' ' if temp_chunk else ''
                        temp_chunk += sentence
                
                if temp_chunk:
                    chunks.append(temp_chunk)
                current_chunk = ""
        else:
            # Paragraph fits, add it
            current_chunk += ' ' if current_chunk else ''  # Changed from '\n\n' to ' '
            current_chunk += paragraph
    
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
