import argparse
import glob
import os
from pathlib import Path

import cv2
import tensorflow as tf
import dask.dataframe as dd
from waymo_open_dataset import v2


def read(tag: str, context_name) -> dd.DataFrame:
    paths = tf.io.gfile.glob(f"{dataset_dir}/{tag}/{context_name}.parquet")
    return dd.read_parquet(paths)


def parse_for_scene(context_name):
    try:
        context_name = Path(context_name).stem

        # lidar boxes
        lidar_box_df = read("lidar_box", context_name)
        lidar_box_df = lidar_box_df[lidar_box_df["[LiDARBoxComponent].type"] == 1]

        # lidar box projections
        lidar_image_projection_df = read("projected_lidar_box", context_name)
        lidar_image_projection_df = lidar_image_projection_df[
            lidar_image_projection_df["key.camera_name"] == camera_id
        ]

        # camera images
        camera_image_df = read("camera_image", context_name)
        camera_image_df = camera_image_df[
            camera_image_df["key.camera_name"] == camera_id
        ]

        data_df = v2.merge(
            v2.merge(lidar_box_df, lidar_image_projection_df), camera_image_df
        )

        # Save full images
        for _, row in camera_image_df.iterrows():
            camera_image = v2.CameraImageComponent.from_dict(row)
            full_img = tf.image.decode_jpeg(camera_image.image).cpu().numpy()
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            full_image_dir = f"{export_dir}/{context_name}"
            os.makedirs(full_image_dir, exist_ok=True)
            cv2.imwrite(
                f"{full_image_dir}/{camera_image.key.frame_timestamp_micros}.jpeg",
                full_img,
            )

        data_df = (
            data_df.groupby(["key.segment_context_name", "key.laser_object_id"])
            .agg(list)
            .reset_index()
        )

        # Save Individual Objects
        for _, row in data_df.iterrows():
            projection_box = v2.ProjectedLiDARBoxComponent.from_dict(row)
            camera_image = v2.CameraImageComponent.from_dict(row)
            lidar_box = v2.LiDARBoxComponent.from_dict(row)

            num_images_for_obj = len(projection_box.type)

            is_small_obj = True
            image_names = []
            images = []
            headings = []

            # Save cropped images
            for idx in range(num_images_for_obj):
                # Only save "bigger" images for labeling
                if (
                    projection_box.box.size.x[idx] * projection_box.box.size.y[idx]
                    > 1000
                ):
                    is_small_obj = False

                img = tf.image.decode_jpeg(camera_image.image[idx]).cpu().numpy()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = img[
                    int(
                        projection_box.box.center.y[idx]
                        - 0.5 * projection_box.box.size.y[idx]
                    ) : int(
                        projection_box.box.center.y[idx]
                        - 0.5 * projection_box.box.size.y[idx]
                        + projection_box.box.size.y[idx]
                    ),
                    int(
                        projection_box.box.center.x[idx]
                        - 0.5 * projection_box.box.size.x[idx]
                    ) : int(
                        projection_box.box.center.x[idx]
                        - 0.5 * projection_box.box.size.x[idx]
                        + projection_box.box.size.x[idx]
                    ) :,
                ]
                image_names.append(
                    f"{export_dir}/{context_name}/{projection_box.key.laser_object_id}/{camera_image.key.frame_timestamp_micros[idx]}.jpeg"
                )
                images.append(img)
                headings.append(
                    (
                        camera_image.key.frame_timestamp_micros[idx],
                        lidar_box.box.heading[idx],
                    )
                )

            if is_small_obj is False:
                image_dir = (
                    f"{export_dir}/{context_name}/{projection_box.key.laser_object_id}"
                )
                os.makedirs(image_dir, exist_ok=True)

                for idx in range(len(images)):
                    cv2.imwrite(image_names[idx], images[idx])

                with open(f"{image_dir}/info.txt", "w+") as f:
                    for annotation in headings:
                        f.write(
                            f"{str(annotation[0])} {str(annotation[1])} {os.linesep}"
                        )

                if num_images_for_obj != len(os.listdir(image_dir)) - 1:
                    raise Exception(
                        f"for scene: and object: not all images were saved!"
                    )
    except Exception as e:
        print(e)
        print(
            f"fatal issue for scene: {context_name} and object: {projection_box.key.laser_object_id}"
        )


if __name__ == "__main__":
    camera_id = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("export_dir", type=str)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    export_dir = f"{args.export_dir}/img"

    scenes = sorted(glob.glob(f"{dataset_dir}/camera_box/*.parquet"))
    scene_num = len(scenes)
    from multiprocessing import Pool
    from tqdm.auto import tqdm

    processes = 8
    with Pool(processes) as pool:
        with tqdm(total=scene_num) as pbar:
            for _ in pool.imap_unordered(parse_for_scene, scenes):
                pbar.update()
