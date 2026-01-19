# Higgs-Audio 文本到语音生成使用示例

本文档提供了使用Higgs-Audio从文本生成语音的详细示例。

## 1. 基本用法

### 使用默认的zh_text示例

```bash
cd examples/serve_engine
python run_hf_example.py zh_text
```

这将使用`zh.txt`文件中的中文文本生成语音，并保存为`output_zh_text.wav`。

### 使用自定义文本文件

```bash
python run_hf_example.py zh_text --text-file /path/to/your/text/file.txt
```

## 2. 使用独立脚本

我们提供了一个独立的脚本`text_to_speech.py`，专门用于从文本生成语音（支持长文本自动分段拼接、可选变速）：

```bash
python text_to_speech.py --text-file zh.txt --output-file output_custom.wav
```

## 3. 文本预处理功能

我们的脚本包含了文本预处理功能，可以自动处理中英文标点符号的转换：

- 中文标点符号（如：，。：；？！）会被转换为对应的英文标点符号
- 特殊标签（如[laugh]、[music]等）会被转换为相应的SE标签
- 多余的空白字符会被清理

## 4. 高级用法

### 调整生成参数

您可以通过修改脚本中的参数来调整生成效果：

```python
output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=input_sample,
    max_new_tokens=2048,  # 增加生成长度
    temperature=0.7,      # 调整随机性
    top_p=0.95,  # 调整生成质量，稳定性较高
    top_k=40, # 调整生成质量，稳定性较高
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
)
```

### 使用不同的语音特征

您可以修改system prompt来指定不同的语音特征：

```python
system_prompt = (
    "Generate audio following instruction.\n\n"
    "<|scene_desc_start|>\n"
    "SPEAKER0: clear voice; moderate pitch; neutral tone\n"  # 修改这里
    "<|scene_desc_end|>"
)
```

## 5. 故障排除

### 模型路径问题

如果遇到模型加载错误，请确保：
1. 模型路径正确
2. 或者能访问HuggingFace Hub

脚本会自动检测本地模型路径是否存在，如果不存在则使用HuggingFace Hub。

### 生成质量问题

如果生成的音频质量不佳，可以尝试：
1. 调整temperature参数（通常在0.6-0.8之间效果较好）
2. 调整top_p参数
3. 确保输入文本没有过多的特殊字符

### 长文本处理

对于较长的文本：
1. 脚本会自动按句子边界分段逐段生成，并在末尾拼接成一段音频（默认每段约 120 字符）
2. 如需更大/更小段落，可通过 `--chunk-max-chars` 调整；设为 0 可禁用分段
3. 如需让拼接更自然，可使用 `--silence-ms` 插入短静音，或用 `--crossfade-ms` 做交叉淡入淡出

## 6. 示例输出

运行成功后，您将看到类似以下的输出：

```
2026-01-19 14:00:00.000 | INFO     | __main__:main:30 - Using device: cuda
2026-01-19 14:00:00.000 | INFO     | __main__:main:40 - Starting generation...
2026-01-19 14:00:30.000 | INFO     | __main__:main:50 - Generation time: 30.00 seconds
2026-01-19 14:00:30.000 | INFO     | __main__:main:53 - Generated text:
一些生成的文本内容...
2026-01-19 14:00:30.000 | INFO     | __main__:main:54 - Saved audio to output_zh_text.wav
```

生成的音频文件将保存在当前目录下。