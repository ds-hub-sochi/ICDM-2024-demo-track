root = '/home/jovyan/murtazin/datasets/WLASL/wlasl_raw/videos'
# Open the input CSV file for reading and a new CSV file for writing
input_csv = '/home/jovyan/murtazin/datasets/WLASL/wlasl_kinetic_format/test.csv'
output_csv = '/home/jovyan/murtazin/datasets/WLASL/maev1_format/test.csv'
with open(input_csv, 'r') as input_file, open(output_csv, 'w') as output_file:
    # Iterate through each line in the input file
    for line in input_file:
        # Add "path/to" to the beginning of each line and write it to the output file
        modified_line = root + '/' + line
        output_file.write(modified_line)

# Close both files
input_file.close()
output_file.close()