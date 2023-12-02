import argparse
import glob
import os
import subprocess
from multiprocessing import Pool

import imagesize
import tqdm


def get_max_dimensions(directory):
    max_width = 0
    max_height = 0

    for filename in glob.glob(f"{directory}/*.jpeg"):
        file_width, file_height = imagesize.get(filename)

        if file_width > max_width:
            max_width = file_width

        if file_height > max_height:
            max_height = file_height

    return max_width + 1, max_height + 1


def generate_videos_for_scene(scene_dir):
    scene_path_in = f"{dataset_dir}/{scene_dir}"
    scene_path_out = f"{export_dir}/{scene_dir}"
    os.makedirs(scene_path_out, exist_ok=True)

    # Scene Images
    scene_video_path = os.path.join(f"{export_dir}/{scene_dir}", "scene.mp4")
    ffmpeg_cmd = f'ffmpeg -y -pattern_type glob -i "{os.path.join(scene_path_in, "*.jpeg")}" -s 960x640 -c:v libx264 -crf 15 -preset fast -c:a aac -b:a 64k -r 10 "{scene_video_path}"'
    complete = subprocess.run(
        ffmpeg_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
    )
    if complete.returncode != 0:
        print(f"{complete.returncode} for scene {scene_path_in}")

    # Individual Object Images
    for object_dir in os.listdir(scene_path_in):
        object_path_in = f"{scene_path_in}/{object_dir}"
        object_path_out = f"{scene_path_out}/{object_dir}"
        if not os.path.isdir(object_path_in):
            continue

        os.makedirs(f"{object_path_out}", exist_ok=True)

        max_width, max_height = get_max_dimensions(object_path_in)
        video_out = f"{object_path_out}/scene.mp4"
        ffmpeg_cmd = f'ffmpeg -y -framerate 1 -pattern_type glob -i "{object_path_in}/*.jpeg" -vf "pad={max_width}:{max_height}:(ow-iw)/2:(oh-ih)/2" -r 30 "{video_out}"'
        complete = subprocess.run(
            ffmpeg_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        if complete.returncode != 0:
            print(f"{complete.returncode} for object {object_path_in}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("export_dir", type=str)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    export_dir = f"{args.export_dir}/vid"

    processes = 8
    with Pool(processes) as pool:
        scene_dirs = os.listdir(dataset_dir)

        with tqdm.tqdm(total=len(scene_dirs)) as pbar:
            for _ in pool.imap_unordered(generate_videos_for_scene, scene_dirs):
                pbar.update()
