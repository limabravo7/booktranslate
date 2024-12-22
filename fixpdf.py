import argparse
import json
from pathlib import Path
import fitz  # PyMuPDF
import openai
import re
import subprocess
import yaml

def read_fix_pages(fix_pages_path):
    """Read the list of pages to be fixed from the fixpdfpages.txt file"""
    with open(fix_pages_path, 'r') as file:
        pages = file.read().strip().split(',')
        return [int(page.strip()) for page in pages]

def collect_input_pages(fixjobpath, pages_to_fix):
    """Collect HTML content of pages to be fixed"""
    input_pages = []
    for page_num in pages_to_fix:
        page_file = fixjobpath / f'page_{page_num:04d}.html'
        if page_file.exists():
            with open(page_file, 'r', encoding='utf-8') as file:
                html_content = file.read()
                input_pages.append((f'{page_num:04d}', html_content))
        else:
            print(f"Warning: Page file not found: {page_file}")
    return input_pages

def save_input_pages(fixjobpath, input_pages):
    """Save the collected input pages to input_pages.json"""
    input_pages_path = fixjobpath / 'input_pages.json'
    with open(input_pages_path, 'w', encoding='utf-8') as file:
        json.dump(input_pages, file, indent=2)
    print(f"Saved input pages to {input_pages_path}")

def translate_pages(client, input_pages, system_message, fixjobpath, model):
    """Translate each page in input_pages via OpenAI and save to fixed_pages.json"""
    fixed_pages = {}
    fixed_pages_path = fixjobpath / 'fixed_pages.json'
    total_pages = len(input_pages)
    for index, (page_num, html_content) in enumerate(input_pages):
        remaining_pages = total_pages - index
        print(f"Translating page {page_num} ({remaining_pages} pages remaining)")
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": html_content}
            ]
        )
        translated_html = response.choices[0].message.content
        translated_html = re.sub(r'^```[a-zA-Z]*\n', '', translated_html)
        translated_html = re.sub(r'```$', '', translated_html)
        fixed_pages[page_num] = translated_html
        
        # Save fixed_pages to fixed_pages.json
        with open(fixed_pages_path, 'w', encoding='utf-8') as file:
            json.dump(fixed_pages, file, indent=2)
        print(f"Saved fixed page {page_num} to {fixed_pages_path}")
    return fixed_pages

def replace_pdf_pages(input_pdf_path, fixed_pages, output_pdf_path):
    """Replace the corresponding pages in the PDF file with the fixed pages"""
    doc = fitz.open(input_pdf_path)
    page_width = 6.5 * 72  # inches to points
    page_height = 9.5 * 72
    margin = 0.5 * 72

    for page_num, translated_html in fixed_pages.items():
        page_index = int(page_num)
        doc.delete_page(page_index)  # Delete the existing page
        page = doc.new_page(pno=page_index, width=page_width, height=page_height)  # Insert a new page
        page.insert_htmlbox(fitz.Rect(margin, margin, page_width - margin, page_height - margin), translated_html)
    doc.save(output_pdf_path)
    print(f"Fixed PDF saved to {output_pdf_path}")

def compress_pdf(input_pdf_path, output_pdf_path):
    """Compress PDF using Ghostscript with font consolidation"""
    try:
        subprocess.run([
            'gs',
            '-sDEVICE=pdfwrite',
            '-dCompatibilityLevel=1.4',
            '-dPDFSETTINGS=/screen',
            '-dSubsetFonts=true',
            '-dEmbedAllFonts=true',
            '-dCompressFonts=true',
            '-dNOPAUSE',
            '-dQUIET',
            '-dBATCH',
            f'-sOutputFile={output_pdf_path}',
            input_pdf_path
        ], check=True)
        print(f"Compressed PDF saved to {output_pdf_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error compressing PDF: {e}")

def read_config():
    """Read the OpenAI API key from config.yaml"""
    config_file = 'config.yaml'
    try:
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config
    except FileNotFoundError:
        print(f"Error: {config_file} not found. Please create it with your OpenAI API key.")
        print("Example config.yaml content:\nopenai:\n  api_key: 'your-api-key-here'")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description='Fix specific pages in a PDF translation job.')
    parser.add_argument('--fixjobpath', required=True, help='Path to the job directory containing HTML files.')
    parser.add_argument('--fixinput', required=True, help='Path to the input PDF file.')
    parser.add_argument('--model', default='gpt-4o-mini', help='The OpenAI model to use for translation.')
    args = parser.parse_args()

    fixjobpath = Path(args.fixjobpath)
    fixinput = Path(args.fixinput)
    fix_pages_path = fixjobpath / 'fixpdfpages.txt'
    fixpdfprompt_path = Path('./fixpdfprompt.txt')
    fixed_pages_path = fixjobpath / 'fixed_pages.json'

    if not fixpdfprompt_path.exists():
        print(f"Error: fixpdfprompt.txt not found")
        return

    with open(fixpdfprompt_path, 'r', encoding='utf-8') as file:
        system_message = file.read()

    if fixed_pages_path.exists():
        print(f"Using existing fixed pages from {fixed_pages_path}")
        with open(fixed_pages_path, 'r', encoding='utf-8') as file:
            fixed_pages = json.load(file)
        pages_to_fix = [int(page_num) for page_num in fixed_pages.keys()]
    else:
        if not fix_pages_path.exists():
            print(f"Error: fixpdfpages.txt not found in {fixjobpath}")
            return
        pages_to_fix = read_fix_pages(fix_pages_path)
        input_pages = collect_input_pages(fixjobpath, pages_to_fix)
        save_input_pages(fixjobpath, input_pages)

        config = read_config()
        client = openai.OpenAI(api_key=config['openai']['api_key'])
        fixed_pages = translate_pages(client, input_pages, system_message, fixjobpath, args.model)

    output_pdf_path = fixinput.with_name(f"{fixinput.stem}_fixed.pdf")
    replace_pdf_pages(fixinput, fixed_pages, output_pdf_path)

    compressed_pdf_path = fixinput.with_name(f"{fixinput.stem}_fixed_compressed.pdf")
    compress_pdf(output_pdf_path, compressed_pdf_path)

if __name__ == "__main__":
    main()
