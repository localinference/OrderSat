import torch
from torch import nn


class TinySeq2SeqTransformer(torch.nn.Module if nn is not None else object):
    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        d_model: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        ffn_dim: int,
        dropout: float,
        max_source_positions: int,
        max_target_positions: int,
    ) -> None:
        super().__init__()
        if nn is None:
            raise SystemExit("PyTorch is required to build the model.")

        embedding_vocab_size = vocab_size + 1
        self.pad_id = pad_id
        self.d_model = d_model
        self.token_embedding = nn.Embedding(
            embedding_vocab_size,
            d_model,
            padding_idx=pad_id,
        )
        self.source_position_embedding = nn.Embedding(
            max_source_positions,
            d_model,
        )
        self.target_position_embedding = nn.Embedding(
            max_target_positions,
            d_model,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        source_padding_mask = attention_mask.eq(0)
        target_padding_mask = decoder_input_ids.eq(self.pad_id)

        source_embeddings = self._embed_source(input_ids)
        target_embeddings = self._embed_target(decoder_input_ids)
        target_mask = self._causal_mask(
            length=decoder_input_ids.size(1),
            device=decoder_input_ids.device,
        )
        memory = self.encoder(
            source_embeddings,
            src_key_padding_mask=source_padding_mask,
        )
        hidden = self.decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=target_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=target_padding_mask,
            memory_key_padding_mask=source_padding_mask,
        )

        return self.output_projection(hidden)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_padding_mask = attention_mask.eq(0)
        source_embeddings = self._embed_source(input_ids)
        memory = self.encoder(
            source_embeddings,
            src_key_padding_mask=source_padding_mask,
        )
        return memory, source_padding_mask

    def decode_step(
        self,
        *,
        decoder_input_ids: torch.Tensor,
        memory: torch.Tensor,
        source_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_embeddings = self._embed_target(decoder_input_ids)
        target_mask = self._causal_mask(
            length=decoder_input_ids.size(1),
            device=decoder_input_ids.device,
        )
        hidden = self.decoder(
            tgt=target_embeddings,
            memory=memory,
            tgt_mask=target_mask,
            tgt_is_causal=True,
            tgt_key_padding_mask=decoder_input_ids.eq(self.pad_id),
            memory_key_padding_mask=source_padding_mask,
        )
        return self.output_projection(hidden)

    def _embed_source(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)
        token_embeddings = self.token_embedding(token_ids) * (self.d_model**0.5)
        embeddings = token_embeddings + self.source_position_embedding(positions)
        return self.dropout(embeddings)

    def _embed_target(self, token_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)
        token_embeddings = self.token_embedding(token_ids) * (self.d_model**0.5)
        embeddings = token_embeddings + self.target_position_embedding(positions)
        return self.dropout(embeddings)

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones((length, length), device=device, dtype=torch.bool),
            diagonal=1,
        )
