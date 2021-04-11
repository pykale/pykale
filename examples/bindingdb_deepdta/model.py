from kale.embed.seq_nn import CNNEncoder
from kale.pipeline.deep_dti import DeepDTATrainer
from kale.predict.decode import MLPDecoder


def get_model(cfg):
    # ---- encoder hyper-parameter ----
    num_drug_embeddings = cfg.MODEL.NUM_SMILE_CHAR
    num_target_embeddings = cfg.MODEL.NUM_ATOM_CHAR
    drug_dim = cfg.MODEL.DRUG_DIM
    target_dim = cfg.MODEL.TARGET_DIM
    drug_length = cfg.MODEL.DRUG_LENGTH
    target_length = cfg.MODEL.TARGET_LENGTH
    num_filters = cfg.MODEL.NUM_FILTERS
    drug_filter_length = cfg.MODEL.DRUG_FILTER_LENGTH
    target_filter_length = cfg.MODEL.TARGET_FILTER_LENGTH

    drug_encoder = CNNEncoder(
        num_embeddings=num_drug_embeddings,
        embedding_dim=drug_dim,
        sequence_length=drug_length,
        num_kernels=num_filters,
        kernel_length=drug_filter_length,
    )

    target_encoder = CNNEncoder(
        num_embeddings=num_target_embeddings,
        embedding_dim=target_dim,
        sequence_length=target_length,
        num_kernels=num_filters,
        kernel_length=target_filter_length,
    )

    # ---- decoder hyper-parameter ----
    decoder_in_dim = cfg.MODEL.MLP_IN_DIM
    decoder_hidden_dim = cfg.MODEL.MLP_HIDDEN_DIM
    decoder_out_dim = cfg.MODEL.MLP_OUT_DIM
    dropout_rate = cfg.MODEL.MLP_DROPOUT_RATE

    decoder = MLPDecoder(
        in_dim=decoder_in_dim, hidden_dim=decoder_hidden_dim, out_dim=decoder_out_dim, dropout_rate=dropout_rate
    )

    # ---- learning rate ----
    lr = cfg.SOLVER.LR

    model = DeepDTATrainer(drug_encoder, target_encoder, decoder, lr, **cfg.MODEL, **cfg.SOLVER)

    return model
