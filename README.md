# PDF and EPUB Book Translator using ChatGPT

## Overview
A tool to translate PDF and EPUB books using OpenAI's ChatGPT. Can be used for large documents (e.g., books) because it chunks them to stay under the token limit for each job. Supports batch processing, resuming jobs, and generating bilingual PDFs. You need a funded OpenAI account and API key.

This was inspired by and initially based on [jb41/translate-book](https://github.com/jb41/translate-book) and [KeinNiemand's fork](https://github.com/KeinNiemand/translate-book/). Created with a *lot* of help from Github Copilot.

## Features
These are available from the command line:
- Language settings
- Processing modes for **EPUB**: regular mode processes translation sequentially; batch mode is half the cost but can take up to 24h (usually within an hour); you can change the model used (defaults to `gpt-4o-mini`)
- Resume functionality for **EPUB**: both regular and batch mode jobs can be resumed if interrupted. A regular mode job can be resumed in batch mode, and vice versa.
- Features for **PDF**: we turn PDF pages into images and use the OpenAI Vision API for OCR and translation in a single step. Unfortunately there is no batch API available. Also, we use `gpt-4o` because it is actually cheaper than `gpt-4o-mini` for images. We can however output a bilingual PDF with facing translation.
- **NOTE**: PDF prompts are inconsistent for me. About 10% of the time it only does OCR and refuses to translate. This might not be an issue with less complicated prompts. I recommend running PDF jobs with the `--debug` flag to save temp files. If you encounter untranslated pages, create text files `fixpdfpages.txt` (list of page numbers that need to be fixed) and `fixpdfprompt.txt` inside the job's temp dir, then run `python fixpdf.py --fixjobpath ./temp/[jobdir] --fixinput path/to/pdf-to-be-fixed.pdf`. It will take each corresponding `page_0000.html` and translate it using the prompt in `fixpdfprompt.txt`, and then replace the corresponding page in `pdf-to-be-fixed.pdf` with the new translations. Some pages `gpt-4o-mini` simply refuses to translate. You can try using `gpt-4o` with the `--model` argument.

Other features:
- For EPUBs, you can supply your own prompt for the model in `customepubprompt.txt`. The default prompt is included in that file. You will probably want to tweak it if you're not translating texts on classical antiquity.
- For PDFs, you can also supply your own prompt (remember this is for both OCR and translation) in `custompdfprompt.txt`. Again, you'd probably want to tweak it for your use case.

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/huperaisan/booktranslate.git
    cd booktranslate
    ```
2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```
3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
4. Install Ghostscript:
    - On macOS:
        ```sh
        brew install ghostscript
        ```
    - On Ubuntu:
        ```sh
        sudo apt-get install ghostscript
        ```
    - On Windows:
        Download and install from [Ghostscript](https://www.ghostscript.com/download/gsdnld.html).

## Configuration
Create a `config.yaml` file with your OpenAI API key:
```yaml
openai:
  api_key: 'your-api-key-here'
```

Also note the `customepubprompt.txt` and `custompdfprompt.txt` as mentioned under "Features" above.

## Usage
### For EPUBs
```sh
python booktrans.py --input path/to/book.epub
```
This is the minimal command, and uses these default options: sequential processing (not batch), from DE to EN, using the gpt-4o-mini model, outputs as `./output/book_EN_gpt-4o-mini.epub`.

```sh
python booktrans.py --input path/to/book.epub --output path/to/output.epub --from-lang DE --to-lang EN --model gpt-4o
```
This specifies custom output, languages, and model.

```sh
python booktrans.py --input path/to/book.epub --mode batch
```
This _starts_ a job using batch processing. The script first batch up the translation job and uploaded to the OpenAI account, then waits 5 seconds to check whether batch has started processing, and if it has, the script saves job status and exits. Run the next command to check on batch processing status and complete the output. If you see that the job has "failed" in this step, this means that you've exceeded your batch token limit and will need to either use a different model or break your job down into smaller pieces.

```sh
python booktrans.py --input path/to/book.epub --mode batchcheck
```
After you've started a batch job, use this to check on its status. It will return the status and exit if it's not completed. If the batch has been completed, it downloads the results and writes the output EPUB file. If your job has timed out ("expired" state if not completed in 24h) or you have manually "cancelled" it, the script will check if any part of the batch has been completed and download the partial results. These partial results can be used to **resume** (see below).

```sh
python booktrans.py --input path/to/book.epub --mode resume
```
If a run is incomplete for any reason (e.g., a batch was partially finished, or you manually exited a sequential run), the completed parts are saved in the `./temp` dir and can be resumed. The above commands lists resumable jobs and you can choose one to continue, using sequential (not batch) processing to finish the untranslated chunks. If you want to use batch processing to finish translation, use `--mode resumebatch`.

### For PDFs
```sh
python booktrans.py --input path/to/book.pdf --output path/to/output.pdf --from-lang DE --to-lang EN --mode pdfbilingual
```
These are all the options for PDFs. Only `--input path/to/book.pdf` is required. You cannot change the processing mode because batch mode is not supported. The script will do its best to replicate the layout of the original PDF, but because translation happens page-by-page, broken sentences at the beginning and end of pages may be badly translated. The `pdfbilingual` mode produces alternating pages of original language and corresponding translation (original coming first). No other `--mode` has any effect, and there is currently no resume functionality.

### Options
- `--input`: Input file path (required)
- `--output`: Output file path (optional, defaults to `[input]_[to-lang]_[model].[ext]`)
- `--from-lang`: Source language (default: DE)
- `--to-lang`: Target language (default: EN)
- `--debug`: Keep batch files and temp directory for debugging
- `--mode`: Processing mode (`batch`, `resume`, `test`, `batchcheck`, `resumebatch`, `pdfbilingual`)
- `--model`: The model to use for translation (default: gpt-4o-mini)

## License
This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE.md) file for details.