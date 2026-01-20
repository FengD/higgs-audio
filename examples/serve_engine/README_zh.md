# 使用Higgs-Audio从文本生成语音

本文档介绍了如何使用Higgs-Audio模型从文本文件（如zh.txt）生成语音。

## 快速开始

### 1. 使用预设的zh_text示例

直接运行以下命令使用默认的zh.txt文件生成语音：

```bash
cd examples/serve_engine
python run_hf_example.py zh_text
```

这将使用`zh.txt`文件中的中文文本生成语音，并保存为`output_zh_text.wav`。

### 2. 使用自定义文本文件

您可以使用任何文本文件生成语音：

```bash
python run_hf_example.py zh_text --text-file /path/to/your/text/file.txt
```

### 3. 使用独立脚本

我们还提供了一个独立的脚本`text_to_speech.py`，专门用于从文本生成语音（支持声音克隆、长文本自动分段拼接、可选变速）：

```bash
python text_to_speech.py --text-file zh.txt --output-file output_custom.wav
```

### 4. 多说话人文本生成语音

对于包含多个说话人的文本，我们提供了专用的多说话人脚本`text_to_speech_multi_speaker.py`：

```bash
python text_to_speech_multi_speaker.py --text-file input_multiuser.txt --output-file output_multi.wav
```

详见[TTS_MULTI_SPEAKER_USAGE.md](TTS_MULTI_SPEAKER_USAGE.md)获取详细使用说明。

## 参数说明

- `--text-file`: 输入文本文件的路径（UTF-8编码）
- `--output-file`: 输出音频文件的路径（默认为output_tts.wav）

## 文本格式要求

- 文件应为UTF-8编码
- 支持中文文本
- 建议文本长度适中，过长的文本可能需要调整`max_new_tokens`参数

## 自定义配置

如果您需要调整生成参数，可以直接修改脚本中的以下参数：

- `max_new_tokens`: 控制生成的最大token数量
- `temperature`: 控制生成的随机性（0.0-1.0）
- `top_p`: nucleus采样参数
- `top_k`: top-k采样参数

## 故障排除

1. 如果遇到模型加载错误，请确保模型路径正确或能访问HuggingFace Hub。
2. 如果生成的音频质量不佳，可以尝试调整temperature和top_p参数。
3. 对于较长的文本，可能需要增加max_new_tokens的值。