import argparse
import base64
import json
import yaml
from pathlib import Path
from pdf_handler import PDFHandler
from openai import OpenAI
import fitz  # PyMuPDF

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

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def sanitize_html_output(html_content):
    """Sanitize the HTML output by removing markdown code block markers"""
    lines = html_content.split('\n')
    if lines[0].strip() == "```html":
        lines = lines[1:]
    if lines[-1].strip() == "```":
        lines = lines[:-1]
    return '\n'.join(lines)

def write_pdf(translations, output_pdf_path):
    """Write a new PDF document from HTML pages"""
    doc = fitz.open()
    page_width = 6 * 72  # 6 inches in points
    page_height = 9 * 72  # 9 inches in points
    margin = 1 * 72  # 1 inch in points

    for chunk_id in sorted(translations.keys(), key=lambda x: int(x.split('-')[1])):
        html_content = translations[chunk_id]
        page = doc.new_page(width=page_width, height=page_height)
        # Insert HTML content into the page with a bounding box
        page.insert_htmlbox(fitz.Rect(margin, margin, page_width - margin, page_height - margin), html_content)
    doc.save(output_pdf_path)
    print(f"PDF saved to {output_pdf_path}")

def main():
    parser = argparse.ArgumentParser(description='Test PDFHandler by generating images of PDF pages and extracting text.')
    parser.add_argument('--input', required=True, help='Input PDF file path.')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for the output images (default: 150).')
    parser.add_argument('--writepdf', action='store_true', help='Write a new PDF document from HTML pages.')
    args = parser.parse_args()

    input_pdf_path = Path(args.input)
    if not input_pdf_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return

    if args.writepdf:
        translations_path = Path('./pdftemp/translations.json')
        if not translations_path.exists():
            print(f"Error: Translations file not found: {translations_path}")
            return

        with open(translations_path, 'r', encoding='utf-8') as f:
            translations = json.load(f)

        output_pdf_path = Path('./output/pdf_imagetest.pdf')
        output_pdf_path.parent.mkdir(exist_ok=True)
        write_pdf(translations, output_pdf_path)
        return

    # Create output directory
    output_dir = Path('./tempimg')
    output_dir.mkdir(exist_ok=True)

    # Generate images of PDF pages
    PDFHandler.generate_page_images(input_pdf_path, output_dir, dpi=args.dpi)

    config = read_config()
    client = OpenAI(api_key=config['openai']['api_key'])

    all_chunks = []
    chapter_map = {}

    # Process each image
    for page_num, image_path in enumerate(sorted(output_dir.glob("*.png"))):
        base64_image = encode_image(image_path)
        print(f"Translating {image_path}")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract the text from this image and produce an HTML document with formatting that reproduces the formatting of the input image. Do not use tables. Pay special attention to ITALICS and SUPERSCRIPT formatting. CRITICAL: If a block of text is smaller than other text in the image, use 'font-size: smaller' for it in your HTML. Use 'font-family: serif' and default line-height. If the image contains GREEK characters, recognise them letter-by-letter and do not try to make sense of the words. Return ONLY the HTML content, beginning with <!DOCTYPE html> and ending with </html>.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ],
                }
            ],
        )

        html_content = response.choices[0].message.content
        sanitized_html_content = sanitize_html_output(html_content)

        # Save HTML content to file
        html_file_path = output_dir / f"page_{page_num}.html"
        with open(html_file_path, "w", encoding="utf-8") as html_file:
            html_file.write(sanitized_html_content)
        
        print(f"Saved HTML: {html_file_path}")

        # Add to all_chunks and chapter_map
        chunk_id = f'chunk-{page_num}'
        all_chunks.append((chunk_id, sanitized_html_content))
        chapter_map[chunk_id] = {
            "item": f"page_{page_num}.html",
            "pos": 0
        }

    # Write all_chunks to all_chunks.json
    pdftemp_dir = Path('./pdftemp')
    pdftemp_dir.mkdir(exist_ok=True)
    with open(pdftemp_dir / 'all_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4)

    # Write chapter_map to chapter_map.json
    with open(pdftemp_dir / 'chapter_map.json', 'w', encoding='utf-8') as f:
        json.dump(chapter_map, f, indent=4)

    print(f"Processing complete. HTML files and metadata saved to {pdftemp_dir}")

if __name__ == "__main__":
    main()
