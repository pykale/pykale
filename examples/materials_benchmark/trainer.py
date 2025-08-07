

    # def layers_freeze(self, mode='all'):
    #     """
    #     Freezes layers of the model based on the provided mode.
    #     Handles LEFTNet, CrystalGraphConvNet, and skips freezing for CartNet.

    #     Args:
    #         mode (str): The freezing mode. Options are:
    #             - 'all': Freeze all layers.
    #             - 'embedding': Freeze only embedding-related layers.
    #             - 'none': Do not freeze any layers.
    #     """
    #     model_name = self.model.__class__.__name__.lower()

    #     # Handle LEFTNet-specific logic
    #     if "leftnet" in model_name:
    #         print(f"Model detected as LEFTNet variant: {self.model.__class__.__name__}")
    #         if mode == 'all':
    #             if hasattr(self.model, 'z_emb'):
    #                 self.model.z_emb.weight.requires_grad = False
    #             if hasattr(self.model, 'radial_emb'):
    #                 for param in self.model.radial_emb.parameters():
    #                     param.requires_grad = False
    #             if hasattr(self.model, 'radial_lin'):
    #                 for param in self.model.radial_lin.parameters():
    #                     param.requires_grad = False
    #             if hasattr(self.model, 'message_layers'):
    #                 for layer in self.model.message_layers:
    #                     for param in layer.parameters():
    #                         param.requires_grad = False
    #             if hasattr(self.model, 'FTEs'):
    #                 for fte in self.model.FTEs:
    #                     for param in fte.parameters():
    #                         param.requires_grad = False

    #         elif mode == 'embedding':
    #             if hasattr(self.model, 'z_emb'):
    #                 self.model.z_emb.weight.requires_grad = False

    #         elif mode == 'none':
    #             print("No layers are frozen.")

    #         else:
    #             raise ValueError("Invalid mode. Choose from 'all', 'embedding', or 'none'.")

    #         print(f'LAYERS FREEZED MODE: {mode.upper()} for {self.model.__class__.__name__}')
    #         return

    #     # Handle CrystalGraphConvNet-specific logic
    #     elif "crystalgraphconvnet" in model_name:
    #         print(f"Model detected as CrystalGraphConvNet: {self.model.__class__.__name__}")

    #         for param in self.model.parameters():
    #             param.requires_grad = True

    #         if mode == 'all' and hasattr(self.model, 'embedding'):
    #             self.model.embedding.weight.requires_grad = False
    #             if hasattr(self.model.embedding, 'bias'):
    #                 self.model.embedding.bias.requires_grad = False
    #             if hasattr(self.model, 'convs'):
    #                 for conv in self.model.convs:
    #                     if hasattr(conv, 'fc_full'):
    #                         conv.fc_full.weight.requires_grad = False
    #                         if hasattr(conv.fc_full, 'bias'):
    #                             conv.fc_full.bias.requires_grad = False
    #                     if hasattr(conv, 'bn1'):
    #                         conv.bn1.weight.requires_grad = False
    #                         conv.bn1.bias.requires_grad = False
    #                     if hasattr(conv, 'bn2'):
    #                         conv.bn2.weight.requires_grad = False
    #                         conv.bn2.bias.requires_grad = False

    #         elif mode == 'embedding' and hasattr(self.model, 'embedding'):
    #             self.model.embedding.weight.requires_grad = False
    #             if hasattr(self.model.embedding, 'bias'):
    #                 self.model.embedding.bias.requires_grad = False

    #         elif mode not in ['all', 'embedding', 'none']:
    #             raise ValueError("Invalid mode. Choose from 'all', 'embedding', or 'none'.")

    #         print(f'LAYERS FREEZED MODE: {mode.upper()} for {self.model.__class__.__name__}')
    #         return

    #     # Skip layer freezing for CartNet
    #     elif "cartnet" in model_name:
    #         print(f"Skipping layer freezing for CartNet: {self.model.__class__.__name__}")
    #         return

    #     # Raise error for invalid models
    #     else:
    #         raise ValueError(f"Invalid model detected: {self.model.__class__.__name__}. "
    #                         f"Expected models are LEFTNet variants, CrystalGraphConvNet, or CartNet.")


# class MetricsCallback(pl.Callback):
#     def __init__(self):
#         self.metrics = []

#     def on_validation_epoch_end(self, trainer, pl_module):
#         metrics = trainer.callback_metrics
#         epoch_metrics = {
#             'epoch': trainer.current_epoch,
#             'val_loss': float(metrics['val_loss']) if 'val_loss' in metrics else None,
#             'val_mae': float(metrics['val_mae']) if 'val_mae' in metrics else None,
#             'val_mre': float(metrics['val_mre']) if 'val_mre' in metrics else None,
#             'val_r2': float(metrics['val_r2']) if 'val_r2' in metrics else None,
#             'train_loss': float(metrics['train_loss']) if 'train_loss' in metrics else None,
#             'train_mae': float(metrics['train_mae']) if 'train_mae' in metrics else None,
#             'train_mre': float(metrics['train_mre']) if 'train_mre' in metrics else None,
#             'train_r2': float(metrics['train_r2']) if 'train_r2' in metrics else None,
#             'train_loss_epoch': float(metrics['train_loss_epoch']) if 'train_loss_epoch' in metrics else None,
#             'train_mae_epoch': float(metrics['train_mae_epoch']) if 'train_mae_epoch' in metrics else None,
#             'train_mre_epoch': float(metrics['train_mre_epoch']) if 'train_mre_epoch' in metrics else None,
#             'train_r2_epoch': float(metrics['train_r2_epoch']) if 'train_r2_epoch' in metrics else None
#         }
#         self.metrics.append(epoch_metrics)

    # def get_metrics_dataframe(self):
    #     return pd.DataFrame(self.metrics)
    
