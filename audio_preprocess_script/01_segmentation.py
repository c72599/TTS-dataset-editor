import glob
import os
import argparse
import subprocess


"""
    Main Function
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_of_datasets", type=str, help="root of dataset")
    parser.add_argument("--length", "-l", default=30*60, help="segment length in second")
    args = parser.parse_args()

    # Initialization
    root_of_datasets = args.root_of_datasets
    command = 'ffmpeg -i "{f_input}" -f segment -segment_time {t_length} -c copy "{d_output}/{f_name}_%03d.wav"'

    # Segment all datasets (grab all folder in root of dataset)
    for dir in glob.iglob(f"{root_of_datasets}/*"):
        dir_name = os.path.split(dir)[-1]

        # Skip folder name start with "Segmented_"
        if dir_name[:10] == "Segmented_":
            continue

        # Initialization
        input_dir = os.path.join(root_of_datasets, dir_name)
        output_dir = os.path.join(root_of_datasets, f"Segmented_{dir_name}")
        os.makedirs(output_dir, exist_ok=True)

        # Collect all the name of audio already segmented previously
        exist_audio = set()
        for file in glob.iglob(f"{output_dir}/*"):
            file_name = os.path.split(file)[-1]
            if file_name[-3:] != "mp3" and file_name[-3:] != "wav":
                continue
            exist_audio.add(file_name[:-8])

        # Start segmentation
        for file in glob.iglob(f"{input_dir}/*"):
            file_name = os.path.split(file)[-1]

            # Skip audio already segmented previously
            if (file_name[-3:] != "mp3" and file_name[-3:] != "wav") or file_name[:-4] in exist_audio:
                continue

            # FFMPEG command
            input_path = os.path.join(input_dir, file_name)
            formated_command = command.format(f_input=input_path,
                                              t_length=args.length,
                                              d_output=output_dir,
                                              f_name=file_name[:-4])
            subprocess.call(formated_command, shell=True)
