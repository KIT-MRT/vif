import os
import imageio
import numpy as np
from PIL import Image


class TbImageLogger:
    def __init__(self, logger, inv_transform) -> None:
        self.logger = logger.experiment
        self.inv_transform = inv_transform
        self.last_epoch = -1
        self.idx = 0

    def log_images(
        self,
        dir,
        images,
        target_indication,
        pred_indication,
        target_heading,
        pred_heading,
        epoch,
        batch_idx,
    ):
        if epoch % 5 != 0:
            return

        if batch_idx > 4:
            return

        if self.last_epoch == epoch:
            self.idx = 0

        images = self.inv_transform(images)
        for b_idx in range(images.shape[0]):
            img_list = []
            for image in images[b_idx]:
                img_gif = image.cpu().numpy().transpose(1, 2, 0)
                img_gif = Image.fromarray((img_gif * 255).astype(np.uint8))
                img_list.append(img_gif)

            os.makedirs(f"{self.logger.log_dir}/{dir}_images/{epoch}", exist_ok=True)
            imageio.mimsave(
                f"{self.logger.log_dir}/{dir}_images/{epoch}/{self.idx+b_idx}_ti{target_indication[b_idx]}_pi{pred_indication[b_idx]}|th{target_heading[b_idx]}_ph{pred_heading[b_idx]}.gif",
                img_list,
            )
        self.idx += images.shape[0]
