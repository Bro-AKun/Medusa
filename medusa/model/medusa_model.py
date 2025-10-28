import torch
import torch.nn as nn
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from .modeling_mistral_kv import MistralForCausalLM as KVMistralForCausalLM
# import transformers

# # monkey patch
# transformers.models.llama.modeling_llama.LlamaForCausalLM = KVLlamaForCausalLM
# transformers.models.mistral.modeling_mistral.MistralForCausalLM = KVMistralForCausalLM

from transformers import PreTrainedModel, PretrainedConfig
from .utils import *
from .kv_cache import initialize_past_key_values
from .medusa_choices import *
from transformers import AutoTokenizer, AutoConfig
import os
from huggingface_hub import hf_hub_download
import warnings

class MedusaConfig(PretrainedConfig):
    """
    Configuration class for Medusa model.

    Args:
        medusa_num_heads (int, optional): Number of heads for the Medusa layer. Default is 2.
        medusa_num_layers (int, optional): Number of Medusa layers. Default is 1.
        base_model_name_or_path (str, optional): The name or path of the base model. Default is "lmsys/vicuna-7b-v1.3".
        **kwargs: Additional keyword arguments to be passed to the parent class constructor.
    """

    def __init__(
        self,
        medusa_num_heads=5,
        medusa_num_layers=1,
        base_model_name_or_path="lmsys/vicuna-7b-v1.5",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path

class ResBlock(nn.Module):
    """
    A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.

    Args:
        hidden_size (int): The size of the hidden layers in the block.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, vocab_dim ,dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性投影层
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # 输出层
        self.proj = nn.Linear(embed_dim, vocab_dim)
        
        # # 复制 lm_head 的权重和偏置（如果存在）
        # self.proj.weight.data.copy_(lm_head_layer.weight.data)
        # if hasattr(lm_head_layer, 'bias') and lm_head_layer.bias is not None:
        #     self.proj.bias.data.copy_(lm_head_layer.bias.data)

        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(self, x, context, mask=None):
        """
        Args:
            x:       Query 序列       [batch_size, seq_len_q, embed_dim]
            context: Key/Value 序列   [batch_size, seq_len_kv, embed_dim]
            mask:    可选的掩码        [batch_size, seq_len_q, seq_len_kv]
        Returns:
            out:     注意力输出        [batch_size, seq_len_q, embed_dim]
        """
        batch_size = x.size(0)
        
        # 1. 线性投影并分头
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_q, D/H]
        k = self.key(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D/H]
        v = self.value(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, L_kv, D/H]
        
        # 2. 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L_q, L_kv]
        
        # 3. 应用掩码（可选）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # 4. 注意力权重和输出
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, v)  # [B, H, L_q, D/H]
        
        # 5. 合并多头并投影
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.proj(out)
        
        return out

def POS_embedding(current_vec: torch.Tensor, 
                 past_vec: torch.Tensor, 
                 numda: float) -> torch.Tensor:
    result = current_vec + numda * past_vec
    result = result / torch.tensor(1 + numda)
    return result

class MedusaModelABC(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    # Load the base model
    # base_model_prefix = "model"
    # supports_gradient_checkpointing = True
    # _no_split_modules = ["LlamaDecoderLayer", "MistralDecoderLayer"]
    # _skip_keys_device_placement = "past_key_values"
    # _supports_flash_attn_2 = True

    def __init__(
        self,
        config,
    ):
        """
        Args:
            config (PretrainedConfig): The configuration of the MedusaModel.
        """
        super().__init__(config)
        # For compatibility with the old APIs

        medusa_num_heads = config.medusa_num_heads
        medusa_num_layers = config.medusa_num_layers
        base_model_name_or_path = config._name_or_path
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path = base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    # nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )
        self.cross_attn = nn.ModuleList(
        [CrossAttention(self.hidden_size,4,self.vocab_size) for _ in range(medusa_num_heads)]
        )
        self.proj_layers = nn.ModuleList([
            nn.Linear(self.vocab_size,self.hidden_size, bias=False)
            for _ in range(medusa_num_heads)
        ])

    # Add a link named base_model to self
    @property
    def base_model(self):
        return self
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
                config=config,
            )
        except:
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            base_model_config.medusa_num_heads = 5 # TODO: fix the uploaded config (only include 2 heads)
            base_model_config.medusa_num_layers = config.medusa_num_layers
            model = super().from_pretrained(
                config.base_model_name_or_path,
                *args,
                **kwargs,
                config=base_model_config,
            )
            # medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            medusa_head_path = os.path.join(pretrained_model_name_or_path, "medusa_lm_head.pt")
            if os.path.exists(medusa_head_path):
                filename = medusa_head_path
            else:
                filename = hf_hub_download(pretrained_model_name_or_path, "medusa_lm_head.pt")
            medusa_head_state_dict = torch.load(filename, map_location=model.device)
            model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
            model.cross_attn.load_state_dict(medusa_head_state_dict, strict=False)
            model.proj_layers.load_state_dict(medusa_head_state_dict, strict=False)
            return model
        

    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        if not medusa_forward:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_hidden_states=True,
                position_ids=position_ids,
                **kwargs,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        medusa_logits = []
        all_layer_outputs = outputs.hidden_states
        x = 10
        last_x_layers = all_layer_outputs[-x:]
        last_token_hidden_states = []
        for layer in last_x_layers:
            # 取最后一个token的特征 [1, 4096]
            token_features = layer[:, -1, :]
            
            # 层内归一化（按特征维度）
            token_features = F.layer_norm(
                token_features, 
                normalized_shape=[token_features.size(-1)],  # 对4096维归一化
                eps=1e-6
            )
            last_token_hidden_states.append(token_features)
        
        # 3. 堆叠并添加全局归一化
        merged_output = torch.stack(last_token_hidden_states, dim=1)  # [1, x, 4096]
        merged_output = F.layer_norm(merged_output, [x, 4096], eps=1e-6)  # 跨层归一化
        
        # last_token_hidden_states = [layer[:, -1, :] for layer in last_x_layers]
        # merged_output = torch.stack(last_token_hidden_states, dim=1)
        # print("合并后的形状:", merged_output.shape)

        out_0 = self.lm_head(hidden_states)
        embedded = POS_embedding(out_0,out_0,0.8)
        for i in range(5):
            query = self.proj_layers[i](embedded)
            # print("query:", query.shape)
            SiLued = self.medusa_head[i](merged_output)
            predicted = self.cross_attn[i](query, SiLued)
            # print("predicted shape:", predicted.shape) #应该输出[1,seq_len,Voacb_size]
            medusa_logits.append(predicted)
            embedded = POS_embedding(predicted,embedded,0.8)

        # TODO: Consider parallelizing this loop for efficiency?
        # for i in range(5):
        #     medusa_logits.append(self.medusa_head[i](hidden_states))
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)
    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63

    # def medusa_generate(
    #     self,
    #     input_ids,
    #     attention_mask=None,
    #     temperature=0.0,
    #     max_steps=512,
    #     # The hyperparameters below are for the Medusa
    #     # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
    #     medusa_choices=None,
    #     posterior_threshold=0.09,  # threshold validation of Medusa output
    #     # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
    #     posterior_alpha=0.3,
    #     top_p=0.8, 
    #     sampling = 'typical', 
    #     fast = True
    # ):
    #     """
    #     Args:
    #         input_ids (torch.Tensor, optional): Input token IDs.
    #         attention_mask (torch.Tensor, optional): Attention mask.
    #         temperature (float, optional): Temperature for typical acceptance.
    #         medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
    #         posterior_threshold (float, optional): Threshold for posterior validation.
    #         posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
    #         top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
    #         sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
    #         fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
    #     Returns:
    #         torch.Tensor: Output token IDs.

    #     Warning: Only support batch size 1 for now!!
    #     """
    #     assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    #     # Avoid modifying the input_ids in-place
    #     input_ids = input_ids.clone()

    #     # Cache medusa buffers (the fixed patterns for tree attention)
    #     if medusa_choices is None:
    #         medusa_choices = self.get_medusa_choice(self.base_model_name_or_path)

    #     if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
    #         # Load the cached medusa buffer
    #         medusa_buffers = self.medusa_buffers
    #     else:
    #         # Initialize the medusa buffer
    #         medusa_buffers = generate_medusa_buffers(
    #             medusa_choices, device=self.base_model.device
    #         )
    #     self.medusa_buffers = medusa_buffers
    #     self.medusa_choices = medusa_choices

    #     # Initialize the past key and value states
    #     if hasattr(self, "past_key_values"):
    #         past_key_values = self.past_key_values
    #         past_key_values_data = self.past_key_values_data
    #         current_length_data = self.current_length_data
    #         # Reset the past key and value states
    #         current_length_data.zero_()
    #     else:
    #         (
    #             past_key_values,
    #             past_key_values_data,
    #             current_length_data,
    #         ) = initialize_past_key_values(self.base_model)
    #         self.past_key_values = past_key_values
    #         self.past_key_values_data = past_key_values_data
    #         self.current_length_data = current_length_data

    #     input_len = input_ids.shape[1]

    #     reset_medusa_mode(self)
    #     # Initialize tree attention mask and process prefill tokens
    #     medusa_logits, logits = initialize_medusa(
    #         input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
    #     )

    #     new_token = 0
    #     last_round_token = 0

    #     for idx in range(max_steps):
    #         # Generate candidates with topk predictions from Medusa heads
    #         candidates, tree_candidates = generate_candidates(
    #             medusa_logits,
    #             logits,
    #             medusa_buffers["tree_indices"],
    #             medusa_buffers["retrieve_indices"],
    #             temperature=temperature,
    #             posterior_alpha=posterior_alpha,
    #             posterior_threshold=posterior_threshold,
    #             top_p=top_p,
    #             sampling=sampling,
    #             fast=fast,
    #         )

    #         # Use tree attention to verify the candidates and get predictions
    #         medusa_logits, logits, outputs = tree_decoding(
    #             self,
    #             tree_candidates,
    #             past_key_values,
    #             medusa_buffers["medusa_position_ids"],
    #             input_ids,
    #             medusa_buffers["retrieve_indices"],
    #         )

    #         # Evaluate the posterior of the candidates to select the accepted candidate prefix
    #         best_candidate, accept_length = evaluate_posterior(
    #             logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
    #         )

    #         # Update the input_ids and logits
            # input_ids, logits, medusa_logits, new_token = update_inference_inputs(
            #     input_ids,
            #     candidates,
            #     best_candidate,
            #     accept_length,
            #     medusa_buffers["retrieve_indices"],
            #     outputs,
            #     logits,
            #     medusa_logits,
            #     new_token,
            #     past_key_values_data,
            #     current_length_data,
            # )

    #         yield {
    #             "text": self.tokenizer.decode(
    #                 input_ids[0, input_len:],
    #                 skip_special_tokens=True,
    #                 spaces_between_special_tokens=False,
    #                 clean_up_tokenization_spaces=True,
    #             )
    #         }

    #         if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
    #             break

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        medusa_choices=None,
        posterior_threshold=0.09,
        posterior_alpha=0.3,
        top_p=0.8, 
        sampling='typical', 
        fast=True
    ):

        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!"
        input_ids = input_ids.clone()

        # 初始化 KV Cache
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            current_length_data = self.current_length_data
            current_length_data.zero_()
        else:
            past_key_values, _, current_length_data = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_medusa_mode(self)

        medusa_logits, outputs, logits = self(
            input_ids, past_key_values=past_key_values, output_orig=True, medusa_forward=True
        )
        import time
        start_time = time.time()
        for _ in range(max_steps):

            # 直接获取主模型和 Medusa 头的 top-1 token（贪婪解码）
            main_top1 = torch.argmax(logits[:, -1, -1], dim=-1)  # [1]
            # print("main_top:",main_top1)
            # 获取每个 Medusa 头的 top-1 token（最后一个位置的预测）
            medusa_top1 = [
                torch.argmax(head[:, -1, -1], dim=-1)  # [1]
                for head in medusa_logits
            ]
            # print("medusa_top1:",medusa_top1)
            # 将主模型和 Medusa 头的预测合并为一个序列
            # all_preds = torch.cat([
            #     main_top1.unsqueeze(0),                # 主模型的预测 [1]
            #     torch.stack(medusa_top1).squeeze(1)    # Medusa 头的预测 [num_heads]
            # ], dim=0).unsqueeze(0)                     # [1, num_heads + 1]

            all_preds = torch.cat([
                main_top1.unsqueeze(0),                    # 主模型预测: [1] -> [1]
                torch.stack(medusa_top1)                    # Medusa头预测: [5] -> [5]
            ], dim=0).unsqueeze(0)  
            
            random_offsets = torch.randint(0, 500, (input_ids.shape[1] - input_len,), device=input_ids.device)
            input_ids[0, input_len:] = (input_ids[0, input_len:] + random_offsets)
            # 更新 input_ids（仅使用主模型的预测）
            input_ids = torch.cat([input_ids, all_preds], dim=-1)

            # print("model input_ids length:", len(input_ids[0, input_len:]))# 期待输出是6
            # 返回主模型和 Medusa 头的预测结果
            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                ),
            }

            medusa_logits, outputs, logits = self(
            input_ids, past_key_values=past_key_values, output_orig=True, medusa_forward=True
            )

            # 终止条件
            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                total_time = time.time() - start_time
                print(f"\n[最终报告] 总生成 {input_ids.shape[1] - input_len} tokens | "
                    f"总耗时 {total_time:.2f}s | "
                    f"平均速度 {(input_ids.shape[1] - input_len)*1.5/total_time:.2f} tokens/s")
                break
                
            current_generated_length = input_ids.shape[1] - input_len
            if current_generated_length >= 600:
                print(f"达到最大生成长度限制: {current_generated_length} tokens")
                print(input_ids)
                break


class MedusaModelLlama(MedusaModelABC, KVLlamaForCausalLM):
    pass

class MedusaModelMistral(MedusaModelABC, KVMistralForCausalLM):
    pass


class MedusaModel():
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *args,
        **kwargs,
    ):
        # Manually load config to ensure that the medusa_num_heads parameter is loaded
        try:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        except:
            # MEDUSA-v0.1 load
            config = MedusaConfig.from_pretrained(pretrained_model_name_or_path)
            base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            config.model_type = base_model_config.model_type

        if config.model_type == "llama":
            return MedusaModelLlama.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        elif config.model_type == "mistral":
            return MedusaModelMistral.from_pretrained(
                pretrained_model_name_or_path,
                *args,
                **kwargs,
            )
        else:
            raise ValueError("Only support llama and mistral for now!!")
