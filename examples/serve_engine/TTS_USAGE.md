# 文本到语音生成工具使用说明

本文档介绍了如何使用`text_to_speech.py`脚本从文本生成语音，支持声音克隆和速度控制功能。

## 基本用法

### 1. 简单文本到语音转换

```bash
cd examples/serve_engine
python text_to_speech.py --text-file zh.txt --output-file output.wav
```

### 2. 带声音克隆的文本到语音转换

```bash
python text_to_speech.py \
  --text-file zh.txt \
  --output-file output_cloned.wav \
  --voice-sample ../voice_prompts/belinda.wav \
  --voice-text "../voice_prompts/belinda.txt"
```

### 3. 调整语音速度

```bash
python text_to_speech.py \
  --text-file zh.txt \
  --output-file output_fast.wav \
  --speed 1.5
```

### 4. 综合使用所有功能

```bash
python text_to_speech.py \
  --text-file zh.txt \
  --output-file output_custom.wav \
  --voice-sample ../voice_prompts/mabaoguo.wav \
  --voice-text "../voice_prompts/mabaoguo.txt" \
  --speed 2.0 \
  --temperature 0.2 \
  --max-tokens 4096
```

## 参数说明

- `--text-file`: 输入文本文件路径（必需）
- `--output-file`: 输出音频文件路径（默认：output_tts.wav）
- `--voice-sample`: 声音样本WAV文件路径（可选，用于声音克隆）
- `--voice-text`: 声音样本对应的文本，可以是文本字符串或文本文件路径（可选，与voice-sample配合使用）
- `--speed`: 语音速度因子，范围0.5-2.0（默认：1.0）
- `--temperature`: 生成温度，控制随机性（默认：0.7）
- `--max-tokens`: 最大生成token数（默认：2048）
- `--chunk-max-chars`: 长文本分段时每段最大字符数（默认：200；设为0可禁用分段）
- `--chunk-min-chars`: 分段最小字符数（默认：20；过短尾段会尽量合并到上一段）
- `--silence-ms`: 分段音频拼接时插入的静音时长（毫秒，默认：20）
- `--save-chunks-dir`: 可选，保存每段生成的 wav 到该目录，便于排查拼接/某段失败问题

## 声音样本列表

以下是一些可用的声音样本：

| 名称 | 语言 | 描述 |
|------|------|------|
| belinda | 英语 | 女声 |
| chadwick | 英语 | 男声 |
| mabaoguo | 中文 | 马保国声音 |
| zh_man_sichuan | 中文 | 四川口音男声 |
| en_man | 英语 | 男声 |
| en_woman | 英语 | 女声 |
| en_donald | 英语 | 建国 |

## 使用示例

### 中文文本生成

```bash
python text_to_speech.py \
  --text-file zh.txt \
  --output-file chinese_speech.wav \
  --voice-sample ../voice_prompts/zh_man_sichuan.wav \
  --voice-text "../voice_prompts/zh_man_sichuan.txt" \
  --speed 1.0
  --temperature 0.1
  --max-tokens 4096

```

### 英文文本生成

```bash
echo "Hello, welcome to our presentation about artificial intelligence and its applications in modern technology." > english_text.txt

python text_to_speech.py \
  --text-file english_text.txt \
  --output-file english_speech.wav \
  --voice-sample ../voice_prompts/en_woman.wav \
  --voice-text "../voice_prompts/en_woman.txt" \
  --speed 1.2
  --temperature 0.4
  --max-tokens 2048
```

## 依赖安装

要使用速度控制功能，需要安装额外的依赖：

```bash
pip install librosa
```

如果没有安装librosa，脚本仍可正常运行，但速度控制功能将不可用，音频将以正常速度保存。

## 注意事项

1. 声音克隆功能需要同时提供声音样本文件（.wav）和对应的文本
2. 速度控制功能需要安装librosa库（`pip install librosa`），否则音频将以正常速度保存
3. 温度参数影响生成的随机性，较低值产生更确定的结果，较高值产生更多样化的结果
4. 如果模型路径未正确设置，脚本会自动使用HuggingFace Hub上的模型
5. 对于长文本，脚本会自动分段逐段生成并在末尾拼接（默认每段约 120 字符），以绕开模型对单次稳定输出时长的限制

### 声音样本问题

确保声音样本文件存在且格式正确：
```bash
# 检查声音样本
ls -la ../voice_prompts/*.wav
```

### 生成质量优化

如果生成的音频质量不佳，可以尝试：
1. 调整temperature参数（通常在0.6-0.8之间效果较好）
2. 增加max-tokens值以允许更长的生成
3. 使用不同的声音样本