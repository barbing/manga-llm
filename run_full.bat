@echo off
conda activate manga-llm
python translate_manga.py --input .\input_pages --output .\output_pages
pause
