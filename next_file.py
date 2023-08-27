import re
import sys


# Define the file path
file_path = 'src/test.py'
num = int(sys.argv[1])

# Read the file content
with open(file_path, 'r') as file:
    content = file.read()

# Use regular expression to find and increment the number
new_content = re.sub(r'(autoencoder_)(\d+)', lambda m: f'{m.group(1)}{num:02}', content)

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.write(new_content)
