import re

# Define the file path
file_path = 'src/test.py'

# Read the file content
with open(file_path, 'r') as file:
    content = file.read()

# Use regular expression to find and increment the number
new_content = re.sub(r'(autoencoder_)(\d+)', lambda m: f'{m.group(1)}{int(m.group(2)) + 1:02}', content)

# Write the modified content back to the file
with open(file_path, 'w') as file:
    file.write(new_content)

old_numbers = re.findall(r'autoencoder_(\d+)', content)
new_numbers = re.findall(r'autoencoder_(\d+)', new_content)


for old_num, new_num in zip(old_numbers, new_numbers):
    print(f'Old number: {old_num}, New number: {new_num}')
    break