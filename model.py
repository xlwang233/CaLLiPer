import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from locationencoder.location_encoder import LocationEncoder

import lightning.pytorch as pl
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

class LocCLIP(nn.Module):
    def __init__(self,
                 # text
                 text_encoder: str,
                 # location
                 location_encoder_hparams: dict,
                 ):
        super().__init__()
        self.text_encoder = text_encoder

        if text_encoder == 'clip':
            pass
        elif text_encoder == "llama2":
            pass
        elif text_encoder == "llama3":
            # self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
            self.txt_enc = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", load_in_4bit=True)

        elif text_encoder == "sentence_transformers":
            # self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.txt_enc = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        else:
            raise ValueError(f"Invalid text encoder: {text_encoder}")

        # The projection layer that projects the text features to the same dimension as the location features
        self.text_projection = nn.Linear(self.txt_enc.config.hidden_size, location_encoder_hparams["dim_output"])

        # set all the weights in the model to be non-trainable
        for param in self.txt_enc.parameters():
            param.requires_grad = False

        self.loc_enc = LocationEncoder(
            location_encoder_hparams["pe_type"],
            location_encoder_hparams["nn_type"],
            hparams=location_encoder_hparams
        ).double()  # double precision for better positional encoding
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        # token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(model_output.size()).float()
        return torch.sum(model_output * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_text(self, text_input):
        """
        text_input: dict with keys "input_ids" and "attention_mask"
        """
        # if hasattr(self, 'tokenizer'):
            # input_ids = self.tokenizer(texts, return_tensors="pt").input_ids
        model_output = self.txt_enc(input_ids=text_input["input_ids"], 
                                attention_mask=text_input["attention_mask"],
                                output_hidden_states=True) # (B, L, D)
        
        if self.text_encoder == "llama3":
            model_output = model_output.hidden_states[-1]
        elif self.text_encoder == "sentence_transformers":
            model_output = model_output.last_hidden_state
        else:
            pass

        model_output = self.mean_pooling(model_output, text_input["attention_mask"])  # (B, D)

        model_output = F.normalize(model_output, p=2, dim=1)  # (B, D)
        # last_hidden_state /= last_hidden_state.norm(dim=-1, keepdim=True)  # (B, L, D)
        # last_hidden_state = last_hidden_state.mean(dim=1)  # (B, D)
        return model_output

    def encode_location(self, coords):
        return self.loc_enc(coords.double())

    def forward(self, coords, text_input):
        """
        coords: torch.Tensor of shape (B, 2)
        text_input: dict with keys "input_ids" and "attention_mask"
        """

        image_features = self.encode_text(text_input)  # (B, D_text)
        image_features = self.text_projection(image_features)  # (B, D)
        location_features = self.encode_location(coords).float() # (B, D)
        # normalized features
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        location_features = location_features / location_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ location_features.t()
        logits_per_location = logits_per_image.t()

        return logits_per_image, logits_per_location
    

class CLIPLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            cache_labels=False,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, logits_per_image, logits_per_coord, output_dict=False):
        device = logits_per_image.device

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_coord, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


class LocCLIPLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = LocCLIP(
            text_encoder=config["model"]["text_encoder"],
            location_encoder_hparams=config["model"]["location_encoder"],
        )

        self.clip_loss = CLIPLoss()
        self.learning_rate = config["training"]["learning_rate"]
        self.weight_decay = config["training"]["weight_decay"]
        self.save_hyperparameters()

    def common_step(self, batch, batch_idx):
        coords = torch.hstack([batch["feature_easting"].unsqueeze(-1), batch["feature_northing"].unsqueeze(-1)])
        # texts = batch["text"]
        # t_points = batch["point"]
        text_input = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
        logits_per_text, logits_per_coord = self.model(coords, text_input)
        return self.clip_loss(logits_per_text, logits_per_coord)
         
    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        exclude = (
            lambda n, p: p.ndim < 2
            or "bn" in n
            or "ln" in n
            or "bias" in n
            or "logit_scale" in n
        )
        include = lambda n, p: not exclude(n, p)

        named_parameters = list(self.model.named_parameters())
        gain_or_bias_params = [
            p for n, p in named_parameters if exclude(n, p) and p.requires_grad
        ]
        rest_params = [
            p for n, p in named_parameters if include(n, p) and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": gain_or_bias_params, "weight_decay": 0.0},
                {
                    "params": rest_params,
                    "weight_decay": self.weight_decay,
                },  # specify in configs/default.yaml
            ],
            lr=self.learning_rate,  # specify in configs/default.yaml
        )

        return optimizer

    def forward(self, lonlats):
        embedding = self.positional_encoder(lonlats)
        return self.neural_network(embedding)

    def test_step(self, batch, batch_idx):
        pass


class DirectPositionEmbeddingDecoder(nn.Module):
    """
    A simple MLP decoder that takes the location embedding as input and output the POI feature.
    """
    def __init__(self, location_embed_dim, feature_embed_dim, hidden_dim=64, dropout_prob=0.5):
        super(DirectPositionEmbeddingDecoder, self).__init__()
        self.fc1 = nn.Linear(location_embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, feature_embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        feature_embeds_pred = self.fc2(x)
        return feature_embeds_pred


class GlobalPositionEncDec(nn.Module):
    def __init__(self, num_poi_classes, dim_feature, loc_enc_params, num_neg_samples):
        super(GlobalPositionEncDec, self).__init__()

        self.num_neg_samples = num_neg_samples
        self.poi_class_embedding = nn.Embedding(num_poi_classes, dim_feature)

        self.loc_encoder = LocationEncoder(
            loc_enc_params["pe_type"],
            loc_enc_params["nn_type"],
            hparams=loc_enc_params
        ).double()  # double precision for better positional encoding

        self.decoder = DirectPositionEmbeddingDecoder(loc_enc_params["dim_output"], dim_feature)        

    def forward(self, pos_poi_type, coords, neg_pois_type):
        """
        pos_poi_type: (B,)
        coords: (B, 2)
        neg_pois_type: (B, 10)
        """
        center_poi_embeds = self.poi_class_embedding(pos_poi_type)  # (B, dim_feature)
        center_poi_loc_embeds = self.loc_encoder(coords)  # (B, dim_output)
        center_poi_loc_embeds = center_poi_loc_embeds.float()
        center_poi_pred_embeds = self.decoder(center_poi_loc_embeds)  # (B, dim_feature)
        
        # negative poi embeddings
        # convert the shape of neg_pois_type from (B, 10) to (B*10,)
        neg_poi_embeds = self.poi_class_embedding(neg_pois_type.view(-1))  # (B*10, dim_feature)
        neg_poi_embeds = neg_poi_embeds.view(-1, self.num_neg_samples, neg_poi_embeds.size(1))  # (B, 10, dim_feature)

        # positive score
        # pos: (B,)
        pos = torch.sum(center_poi_embeds * center_poi_pred_embeds, dim=1, keepdim=False)
        
        # negative sampling
        # center_pred_embed_: shape (B, dim_feature) -> (B, num_neg_sample, embed_dim)
        center_pred_embed_ = center_poi_pred_embeds.unsqueeze(1).expand_as(neg_poi_embeds)
        # neg: (B, num_neg_sample)
        neg = torch.sum(neg_poi_embeds * center_pred_embed_, dim=2, keepdim=False)
   
        pos = torch.log(torch.sigmoid(pos))
        neg = torch.sum(torch.log(torch.sigmoid(-neg)), dim=1, keepdim=False)/self.num_neg_samples

        losses = -(pos + neg)
        loss = losses.mean()

        return loss
        # return center_poi_embeds, center_poi_loc_embeds, neg_poi_embeds