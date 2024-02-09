from transformers import PreTrainedModel, PretrainedConfig

# set path
import sys
from scgpt.model import TransformerModel
from scgpt.loss import masked_mse_loss


class scGPT_config(PretrainedConfig):
    model_type = "scGPT"

    def __init__(
        self,
        vocab_size=50000,
        d_hid=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        pad_value=-2,
        mask_value=-1,
        use_mod=False,
        use_batch_labels=False,
        DSBN=False,
        CLS=False,
        GEPC=False,
        ESC=False,
        n_bins=51,
        padding_idx=0,
        do_mvc=False,
        do_dab=False,
        num_batch_labels=None,
        domain_spec_batchnorm=False,
        ecs_threshold=0.3,
        explicit_zero_prob=False,
        use_generative_training=True,
        use_fast_transformer=True,
        fast_transformer_backend="flash",
        use_sim_decoder=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_hid = d_hid
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.use_mod = use_mod
        self.use_batch_labels = use_batch_labels
        self.num_batch_labels = num_batch_labels
        self.DSBN = DSBN
        self.CLS = CLS
        self.GEPC = GEPC
        self.ESC = ESC
        self.pad_value = pad_value
        self.mask_value = mask_value
        self.n_bins = n_bins
        self.padding_idx = padding_idx
        self.do_mvc = do_mvc
        self.do_dab = do_dab
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.ecs_threshold = ecs_threshold
        self.explicit_zero_prob = explicit_zero_prob
        self.use_generative_training = use_generative_training
        self.use_fast_transformer = use_fast_transformer
        self.fast_transformer_backend = fast_transformer_backend
        self.use_sim_decoder = use_sim_decoder

class scGPT_ForPretraining(PreTrainedModel):
    config_class = scGPT_config

    def __init__(self, config):
        super().__init__(config)
        self.scGPT_backbone = TransformerModel(
            ntoken=config.vocab_size,
            d_model=config.n_embd,
            nhead=config.n_head,
            d_hid=config.d_hid,
            nlayers=config.n_layer,
            padding_idx=config.padding_idx,
            dropout=config.dropout,
            pad_value=config.pad_value,
            do_mvc=self.do_mvc,
            do_dab=self.do_dab,
            use_batch_labels=self.use_batch_labels,
            num_batch_labels=self.num_batch_labels,
            domain_spec_batchnorm=self.domain_spec_batchnorm,
            n_input_bins=config.n_bins,
            ecs_threshold=config.ecs_threshold,
            explicit_zero_prob=config.explicit_zero_prob,
            use_generative_training=config.use_generative_training,
            use_fast_transformer=config.use_fast_transformer,
            fast_transformer_backend=config.fast_transformer_backend,
            use_sim_decoder=config.self.use_sim_decoder,
            n_cls=1,
            pad_token="<pad>",
            input_emb_style="continuous",
            cell_emb_style="cls",
            mvc_decoder_style="inner product",
            pre_norm=False,
        )
        self.criterion = masked_mse_loss

    def forward(
        self,
        pcpt_gene,
        pcpt_expr,
        pcpt_key_padding_mask,
        gen_gene,
        gen_key_padding_mask,
        gen_expr_target,
        CLS=False,
        MVC=False,
        generative_training=True,
    ):
        # generative loss
        output_dict = self.scGPT_backbone(
            pcpt_gene,
            pcpt_expr,
            pcpt_key_padding_mask,
            gen_gene,
            gen_key_padding_mask,
            CLS,
            MVC,
            generative_training,
        )
        gen_expr_preds = output_dict["gen_preds"]
        positions_to_match = ~gen_key_padding_mask
        loss = loss_mse = self.criterion(
            gen_expr_preds, gen_expr_target, positions_to_match
        )

        loss_mvc = None
        if MVC:
            # MVC loss
            loss_mvc = self.criterion(
                output_dict["mvc_output"][:, pcpt_gene.shape[1] :],
                gen_expr_target,
                positions_to_match,
            )
            loss = loss + loss_mvc

        # embed -> expr loss
        previous_cell_embs = output_dict["cell_emb"].detach()
        preds = self.scGPT_backbone(
            pcpt_gene,
            pcpt_expr,
            pcpt_key_padding_mask,
            gen_gene,
            gen_key_padding_mask,
            CLS=False,
            MVC=False,
            input_cell_emb=previous_cell_embs,
            generative_training=generative_training,
        )["gen_preds"]
        loss_gen = self.criterion(preds, gen_expr_target, positions_to_match)
        loss = loss + loss_gen

        return {
            "loss": loss,
            "loss_mse": loss_mse,
            "loss_gen": loss_gen,
            "loss_mvc": loss_mvc,
        }


class scGPT_ForSequenceClassification(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.scGPT_backbone = TransformerModel(config)

    def forward(self):
        raise NotImplementedError


if __name__ == "__main__":
    config = scGPT_config()
    model = scGPT_ForPretraining(config)
    print(model)
    # model = scGPT_ForSequenceClassification(config)
