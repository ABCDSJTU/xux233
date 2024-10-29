import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch.nn.functional as F

model_name = "Llama-2-7b-hf/meta-llama_Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    output_attentions=True,
    use_cache=True
)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

with open("text.txt", "r") as file:
    text = file.read()

inputs = tokenizer(text, return_tensors='pt')
input_ids = inputs['input_ids'].to(device)

# 初始化变量
generated = input_ids
past_key_values = None

# 定义阈值
HIGH_THRESHOLD = 0.7
LOW_THRESHOLD = 0.3

# 记录每个 Token 的状态（'gpu', 'cpu', 'discard'）
token_states = ['gpu'] * input_ids.size(1)

# 记录 CPU 上的 past_key_values（仅存储 'cpu' 状态的 Token）
cpu_past_key_values = None

# 定义步数计数器，用于周期性地将CPU上的past_key_values移回 GPU
step_counter = 0
UPDATE_INTERVAL = 5  # 每隔5步将CPU上的past_key_values移回 GPU

# 在推理过程中不计算梯度，节省内存
with torch.no_grad():
    for _ in range(100):
        step_counter += 1

        # 当达到 UPDATE_INTERVAL 时，进行内存交换
        if step_counter % UPDATE_INTERVAL == 0 and cpu_past_key_values is not None:
            for layer_idx in range(len(cpu_past_key_values)):
                if cpu_past_key_values[layer_idx] is not None:
                    key_cpu, value_cpu = cpu_past_key_values[layer_idx]
                    key_cpu = key_cpu.to(device)
                    value_cpu = value_cpu.to(device)

                    key_gpu, value_gpu = past_key_values[layer_idx]
                    past_key_values[layer_idx] = (
                        torch.cat([key_gpu, key_cpu], dim=2),
                        torch.cat([value_gpu, value_cpu], dim=2)
                    )

            cpu_past_key_values = None
            token_states = ['gpu' if s == 'cpu' else s for s in token_states]

        # 构建 effective_past_key_values
        effective_past_key_values = past_key_values

        outputs = model(
            input_ids=generated[:, -1:],  # 只输入最后一个 Token
            past_key_values=effective_past_key_values,
            use_cache=True,
            output_attentions=True,
        )

        logits = outputs.logits
        next_token_logits = logits[:, -1, :]

        # 使用温度采样
        temperature = 0.7
        next_token_logits = next_token_logits / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # 将下一个 Token 添加到生成的序列中
        generated = torch.cat([generated, next_token_id], dim=-1)
        token_states.append('gpu')  # 新生成的 Token 初始状态为 'gpu'

        # 获取注意力和 past_key_values
        attentions = outputs.attentions
        past_key_values = list(outputs.past_key_values)

        # 处理注意力并根据需要更新 token_states
        for layer_idx, layer_attention in enumerate(attentions):
            attention_last_token = layer_attention[:, :, -1, :]
            attention_scores = attention_last_token.mean(dim=1)
            seq_len = attention_scores.shape[1]

            # 更新token_states的长度
            if len(token_states) < seq_len:
                token_states.extend(['gpu'] * (seq_len - len(token_states)))

            for token_idx in range(seq_len - 1):  # 排除最后一个 Token
                attention_score = attention_scores[0, token_idx].item()

                if attention_score < LOW_THRESHOLD:
                    # 标记为 'discard'
                    token_states[token_idx] = 'discard'
                elif attention_score < HIGH_THRESHOLD:
                    # 标记为 'cpu'
                    token_states[token_idx] = 'cpu'
                else:
                    # 标记为 'gpu'
                    token_states[token_idx] = 'gpu'

        # 根据 token_states 处理 past_key_values
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key, value = past_key_values[layer_idx]

                # 获取序列长度
                seq_len = key.shape[2]

                # 调整 token_states 以匹配当前序列长度
                token_states_layer = token_states[-seq_len:]

                # 创建掩码
                gpu_mask = torch.tensor([s == 'gpu' for s in token_states_layer], device=key.device)
                cpu_mask = torch.tensor([s == 'cpu' for s in token_states_layer], device=key.device)
                discard_mask = torch.tensor([s == 'discard' for s in token_states_layer], device=key.device)

                # 处理 'discard' 状态的 Token，直接移除
                keep_mask = ~discard_mask

                key = key[:, :, keep_mask, :]
                value = value[:, :, keep_mask, :]

                # 分离 'cpu' 状态的 Token
                key_cpu = key[:, :, cpu_mask[keep_mask], :]
                value_cpu = value[:, :, cpu_mask[keep_mask], :]

                # 更新 GPU 上的 past_key_values，仅保留 'gpu' 状态的 Token
                key_gpu = key[:, :, gpu_mask[keep_mask], :]
                value_gpu = value[:, :, gpu_mask[keep_mask], :]

                past_key_values[layer_idx] = (key_gpu, value_gpu)

                # 更新 CPU 上的 past_key_values
                if key_cpu.size(2) > 0:
                    if cpu_past_key_values is None:
                        cpu_past_key_values = [None] * len(past_key_values)
                    cpu_past_key_values[layer_idx] = (key_cpu.cpu(), value_cpu.cpu())

            #更新token_states，只保留 'gpu' 和 'cpu' 状态的 Token
            token_states = [s for s in token_states if s != 'discard']

        # 检查是否生成了结束标记
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    new_tokens = generated[0][input_ids.size(1):]
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))