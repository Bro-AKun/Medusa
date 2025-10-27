import torch


class KVCache:
    """
    A key-value cache for the model.

    This class provides a mechanism to maintain a growing cache of keys and values,
    particularly useful for models that benefit from caching previous states,
    like transformers during autoregressive decoding.

    Attributes:
        data (torch.Tensor): The tensor storing keys and values.
        current_length (int): Current length of the data being stored.
    """

    def __init__(self, data, current_length):
        """
        Initialize the KVCache.

        Args:
            data (torch.Tensor): Initial tensor to store the keys and values.
            current_length (int): Initial length of the data.
        """
        self.data = data
        self.current_length = current_length

    @property
    def shape(self):
        """Return the shape of the data tensor with updated length."""
        return (
            self.data.shape[0],
            self.data.shape[1],
            self.current_length.item(),
            self.data.shape[3],
        )

    def copy(self, indices: torch.Tensor, prev_length: int, dim: int = 2):
        """
        Copy values from the current data at specified indices to a new location.

        Args:
            indices (torch.Tensor): Indices of the data tensor to be copied.
            prev_length (int): Previous length before adding new data.
            dim (int, optional): Dimension along which copying should be performed. Default is 2.
        """
        tgt = self.data.index_select(dim, indices)
        dst = self.data.narrow(dim, prev_length, tgt.shape[dim])
        dst.copy_(tgt, non_blocking=True)
        self.current_length.fill_(prev_length + tgt.shape[dim])

    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.

        Args:
            tensor (torch.Tensor): The tensor to be concatenated.
            dim (int, optional): The dimension along which concatenation should be done. Default is 2.

        Returns:
            torch.Tensor: The data tensor after concatenation up to the current length.
        """
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        self.current_length.add_(tensor.shape[dim])
        return torch.narrow(self.data, 2, 0, self.current_length)
    
    def cat(self, tensor: torch.Tensor, dim: int = 2):
        """
        Concatenate the given tensor with the current data.
        """
        # 获取当前长度和最大长度
        current_len = self.current_length.item() if isinstance(self.current_length, torch.Tensor) else self.current_length
        max_len = self.data.shape[dim]
        
        # 检查是否会超出限制
        if current_len + tensor.shape[dim] > max_len:
            # 动态扩展缓存
            required_additional = current_len + tensor.shape[dim] - max_len
            self._expand_cache(required_additional, dim)
        
        dst = self.data.narrow(dim, self.current_length, tensor.shape[dim])
        dst.copy_(tensor)
        
        # 更新长度
        if isinstance(self.current_length, torch.Tensor):
            self.current_length.add_(tensor.shape[dim])
        else:
            self.current_length += tensor.shape[dim]
        
        return torch.narrow(self.data, dim, 0, self.current_length)

    def _expand_cache(self, additional_size, dim):
        """动态扩展缓存大小"""
        old_size = self.data.shape[dim]
        new_size = old_size + additional_size
        print(f"动态扩展 KV Cache: {old_size} -> {new_size}")
        
        # 创建新的更大缓存
        new_shape = list(self.data.shape)
        new_shape[dim] = new_size
        
        new_data = torch.zeros(
            *new_shape,
            dtype=self.data.dtype,
            device=self.data.device
        )
        
        # 复制现有数据
        slices = [slice(None)] * self.data.dim()
        slices[dim] = slice(0, self.current_length)
        new_data[tuple(slices)] = self.data[tuple(slices)]
        
        self.data = new_data


def initialize_past_key_values(model):
    """
    Initialize past key and value states for a given transformer model.

    This function prepares key-value cache structures for the model, allowing it to store and reuse
    past key and value states during autoregressive decoding, which can improve efficiency.

    Args:
        model (nn.Module): The transformer model for which past key-value states need to be initialized.

    Returns:
        tuple:
            - past_key_values (list): A list of KVCache objects for each layer in the model.
            - past_key_values_data (torch.Tensor): The tensor that will store all keys and values.
            - current_length_data (torch.Tensor): A tensor tracking the current length of keys/values in the cache.
    """
    # Extracting configuration from the model
    config = model.config
    # Initializing the batch size to 1, this can be modified if different batch sizes are required
    batch_size = 1
    # Initializing a tensor to store past keys and values for all layers
    past_key_values_data = torch.zeros(
        config.num_hidden_layers * 2,
        batch_size,
        config.num_key_value_heads,
        config.max_position_embeddings,
        config.hidden_size // config.num_attention_heads,
        device=model.device,
        dtype=model.dtype,
    )
    # Initialize tensor to store the current length of the cached data for all layers.
    # [IMPORTANT] It needs to be kept on CPU for quick access and updates.
    current_length_data = torch.zeros(
        config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
    )
    # Creating a KVCache for each pair of key and value in all layers
    past_key_values = [] * config.num_hidden_layers
    for i in range(config.num_hidden_layers):
        past_key_values.append(
            [
                KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
                for j in range(2)
            ]
        )
    return past_key_values, past_key_values_data, current_length_data

# def initialize_past_key_values(model, max_length_multiplier=2):
#     """
#     Initialize past key and value states for a given transformer model.

#     Args:
#         model (nn.Module): The transformer model.
#         max_length_multiplier (int): 缓存大小倍数，默认为2倍最大位置嵌入
#     """
#     config = model.config
#     batch_size = 1
    
#     # 增加缓存大小
#     cache_size = config.max_position_embeddings * max_length_multiplier
    
#     past_key_values_data = torch.zeros(
#         config.num_hidden_layers * 2,
#         batch_size,
#         config.num_key_value_heads,
#         cache_size,  # 使用更大的缓存大小
#         config.hidden_size // config.num_attention_heads,
#         device=model.device,
#         dtype=model.dtype,
#     )
    
#     current_length_data = torch.zeros(
#         config.num_hidden_layers * 2, dtype=torch.long, device="cpu"
#     )
    
#     past_key_values = [] * config.num_hidden_layers
#     for i in range(config.num_hidden_layers):
#         past_key_values.append(
#             [
#                 KVCache(past_key_values_data[i * 2 + j], current_length_data[i * 2 + j])
#                 for j in range(2)
#             ]
#         )
    
#     print(f"初始化 KV Cache: 最大长度 = {cache_size}")
#     return past_key_values, past_key_values_data, current_length_data
