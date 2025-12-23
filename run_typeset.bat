@echo off
conda activate manga-llm
python translate_manga.py --mode typeset --csv .\output_pages\_extractions.csv --input .\input_pages --output .\output_pages_redraw
pause
