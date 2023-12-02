import torch
import lightning as L
import torchmetrics


class LitVisualIntentionFormer(L.LightningModule):
    def __init__(
        self,
        vif,
        start_lr,
        end_lr,
        img_train_logger=None,
        img_val_logger=None,
        num_ind_classes=3,
        max_epochs=10
    ):
        super().__init__()
        self.vif = vif
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.img_train_logger = img_train_logger
        self.img_val_logger = img_val_logger
        self.ind_classes = num_ind_classes
        self.max_epochs = max_epochs

        self.train_ind_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=self.ind_classes, average="macro"
        )
        self.train_rear_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=3, average="macro"
        )
        self.train_head_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=4, average="macro"
        )

        self.val_ind_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=self.ind_classes, average="macro"
        )
        self.val_rear_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=3, average="macro"
        )
        self.val_head_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=4, average="macro"
        )

    def training_step(self, batch, batch_idx):
        images, headings, targets = batch
        pred_indicator, pred_rear, pred_heading = self.vif(images, headings)

        if self.ind_classes == 3:
            targets_indicator = targets[:, 1:4]
        else:
            targets_indicator = targets[:, :4]
        targets_rear = targets[:, 4:7]
        targets_heading = targets[:, 7:11]

        #
        # Loss
        #
        alpha = 0.5
        gamma = 2.0
        # indicator
        ce_loss = torch.nn.functional.cross_entropy(
            pred_indicator, targets_indicator, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        loss_indicator = (alpha * (1 - pt) ** gamma * ce_loss).mean()
        # rear
        ce_loss = torch.nn.functional.cross_entropy(
            pred_rear, targets_rear, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        loss_rear = (alpha * (1 - pt) ** gamma * ce_loss).mean()
        # heading
        ce_loss = torch.nn.functional.cross_entropy(
            pred_heading, targets_heading, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        loss_heading = (alpha * (1 - pt) ** gamma * ce_loss).mean()

        self.log("train_loss_indicator", loss_indicator, sync_dist=True)
        self.log("train_loss_rear", loss_rear, sync_dist=True)
        self.log("train_loss_heading", loss_heading, sync_dist=True)

        # Metrics
        pred_indicator_labels = pred_indicator.softmax(1).argmax(1)
        target_indicator_labels = targets_indicator.argmax(1)
        self.train_ind_f1(pred_indicator_labels, target_indicator_labels)
        self.log(
            "train_ind_f1",
            self.train_ind_f1,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )

        pred_rear_labels = pred_rear.softmax(1).argmax(1)
        target_rear_labels = targets_rear.argmax(1)
        self.train_rear_f1(pred_rear_labels, target_rear_labels)
        self.log(
            "train_rear_f1",
            self.train_rear_f1,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )

        pred_heading_labels = pred_heading.softmax(1).argmax(1)
        target_heading_labels = targets_heading.argmax(1)
        self.train_head_f1(pred_rear_labels, target_rear_labels)
        self.log(
            "train_head_f1",
            self.train_head_f1,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            prog_bar=True,
        )

        if self.img_train_logger:
            self.img_train_logger.log_images(
                "train",
                images,
                target_indicator_labels,
                pred_indicator_labels,
                target_heading_labels,
                pred_heading_labels,
                self.current_epoch,
                batch_idx,
            )

        return (1 * loss_indicator) + (1 * loss_heading) + (1 * loss_rear)

    def validation_step(self, batch, batch_idx):
        images, headings, targets = batch
        pred_indicator, pred_rear, pred_heading = self.vif(images, headings)

        if self.ind_classes == 3:
            targets_indicator = targets[:, 1:4]  # None not considered!
        else:
            targets_indicator = targets[:, :4]
        targets_rear = targets[:, 4:7]
        targets_heading = targets[:, 7:]

        pred_indicator_labels = pred_indicator.softmax(1).argmax(1)
        target_indicator_labels = targets_indicator.argmax(1)
        self.val_ind_f1(pred_indicator_labels, target_indicator_labels)
        self.log(
            "val_ind_f1",
            self.val_ind_f1,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        pred_heading_labels = pred_heading.softmax(1).argmax(1)
        target_heading_labels = targets_heading.argmax(1)
        self.val_head_f1(pred_heading_labels, target_heading_labels)
        self.log(
            "val_head_f1",
            self.val_head_f1,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        pred_rear_labels = pred_rear.softmax(1).argmax(1)
        targets_rear_labels = targets_rear.argmax(1)
        self.val_rear_f1(pred_rear_labels, targets_rear_labels)
        self.log(
            "val_rear_f1",
            self.val_rear_f1,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        if self.img_val_logger:
            self.img_val_logger.log_images(
                "val",
                images,
                target_indicator_labels,
                pred_indicator_labels,
                target_heading_labels,
                pred_heading_labels,
                self.current_epoch,
                batch_idx,
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.start_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.end_lr
        )
        return [optimizer], [scheduler]
