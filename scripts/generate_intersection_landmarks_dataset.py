import argparse
import sys
import os
import csv


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-paths",
        nargs="+",
        help="input csv folder paths to read the landmarks from",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-path",
        help="output path to save the intersection dataset",
        type=str,
        required=True,
    )
    return parser.parse_args()


def check_fall_files(path):
    fall_file = False
    nofall_file = False

    for file in os.listdir(path):
        file_name = file.split(".")[0]
        if file_name == "fall":
            fall_file = True
        elif file_name == "nofall":
            nofall_file = True

        if fall_file and nofall_file:
            return True
    return False


def get_image_intersection_names(input_files_path):
    intersection_names = []
    for input_file_path in input_files_path:
        image_names = []
        with open(input_file_path, "r", newline="") as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                if line:
                    image_names.append(line[0])
        
        if len(intersection_names):
            intersection_names = list(set(intersection_names).intersection(image_names))
        else:
            intersection_names.extend(image_names)
    return intersection_names


def main():
    try:
        args = cli()
        input_paths, output_path = args.input_paths,args.output_path

        # Check if all folders have fall and nofall files
        for input_path in input_paths:
            if not check_fall_files(input_path):
                print(f"Input path {input_path} does not have fall and nofall files")
        
        # Generate intersection landmarks dataset
        for file_name in ['fall', 'nofall']:
            print(f"Generating intersection for {file_name}")
            
            input_files_path = [os.path.join(input_path, f"{file_name}.csv") for input_path in input_paths]
            image_intersection_names = get_image_intersection_names(input_files_path)
            print(f"Intersection images: {len(image_intersection_names)}")

            for (input_path, input_file_path) in zip(input_paths, input_files_path):
                input_path_folder_name = os.path.basename(input_path)
                output_path_folder = os.path.join(output_path, input_path_folder_name)

                if not os.path.exists(output_path_folder):
                    os.makedirs(output_path_folder)
                
                output_file_path = os.path.join(output_path_folder, f"{file_name}.csv")
                output_lines = []

                with open(input_file_path, "r", newline="") as f:
                    csv_reader = csv.reader(f)
                    for line in csv_reader:
                        if line and line[0] in image_intersection_names:
                            output_lines.append(line)
                
                with open(output_file_path, "w", newline="") as f:
                    csv_writer = csv.writer(f)
                    for output_line in output_lines:
                        csv_writer.writerow(output_line)
        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
