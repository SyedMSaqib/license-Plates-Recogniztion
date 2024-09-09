import re

def remove_comments(code):
    # Remove single-line comments
    code = re.sub(r'#.*', '', code)
    
    # Remove multi-line comments (i.e., docstrings)
    code = re.sub(r'""".*?"""|\'\'\'.*?\'\'\'', '', code, flags=re.DOTALL)
    
    return code

def process_file(input_file, output_file):
    with open(input_file, 'r') as file:
        code = file.read()
    
    cleaned_code = remove_comments(code)
    
    with open(output_file, 'w') as file:
        file.write(cleaned_code)

# Replace 'input_file.py' with your filename and 'output_file.py' with the desired output filename.
input_file = 'LPD.py'
output_file = 'lpd.py'

process_file(input_file, output_file)
