# 多说话人文本到语音生成工具使用说明

本文档介绍了如何使用`text_to_speech_multi_speaker.py`脚本从包含多个说话人的文本生成语音，支持为每个说话人指定不同的声音克隆样本。

## 基本用法

### 1. 多说话人文本到语音转换

```bash
cd examples/serve_engine
python text_to_speech_multi_speaker.py --text-file input_multiuser.txt --output-file output_multi.wav
```

### 2. 指定说话人声音样本

```bash
python text_to_speech_multi_speaker.py \
  --text-file input_multiuser.txt \
  --output-file output_custom_multi.wav \
  --speaker-voices "SPEAKER0:../voice_prompts/belinda.wav,../voice_prompts/belinda.txt" \
  --speaker-voices "SPEAKER1:../voice_prompts/chadwick.wav,../voice_prompts/chadwick.txt"
```

### 3. 调整语音速度

```bash
python text_to_speech_multi_speaker.py \
  --text-file input_multiuser.txt \
  --output-file output_multi_fast.wav \
  --speed 1.2
```

### 4. 使用高质量速度调整

```bash
python text_to_speech_multi_speaker.py \
  --text-file input_multiuser.txt \
  --output-file output_multi_high_quality.wav \
  --speed 1.2 \
  --speed-quality high
```

### 5. 使用低质量速度调整（更快处理）

```bash
python text_to_speech_multi_speaker.py \
  --text-file input_multiuser.txt \
  --output-file output_multi_low_quality.wav \
  --speed 1.2 \
  --speed-quality low
```

## 文本格式要求

多说话人文本需要使用特定的标记格式，每个说话人的文本前需要加上`[SPEAKER*]`标签，其中`*`是数字：

```
[SPEAKER0] 这是第一个说话人的文本内容...

[SPEAKER1] 这是第二个说话人的文本内容...
```

示例文件可以在[examples/serve_engine/input_multiuser.txt](examples/serve_engine/input_multiuser.txt)找到。

## 参数说明

- `--text-file`: 输入文本文件路径（必需）
- `--output-file`: 输出音频文件路径（默认：output_tts_multi.wav）
- `--speaker-voices`: 说话人到声音样本的映射，格式为`SPEAKER*:path/to/voice.wav,path/to/voice.txt`，可以多次使用来指定多个说话人
- `--speed`: 语音速度因子，范围0.5-2.0（默认：1.0）
- `--speed-quality`: 速度调整处理质量，可选值：low、medium、high（默认：high）
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

### 中文多说话人对话生成

```bash
python text_to_speech_multi_speaker.py \
  --text-file input_multiuser.txt \
  --output-file chinese_multi_speech.wav \
  --speaker-voices "SPEAKER0:../voice_prompts/zh_man_sichuan.wav,../voice_prompts/zh_man_sichuan.txt" \
  --speaker-voices "SPEAKER1:../voice_prompts/mabaoguo.wav,../voice_prompts/mabaoguo.txt" \
  --speed 1.0 \
  --temperature 0.1 \
  --max-tokens 4096
```

### 英文多说话人对话生成

```bash
python text_to_speech_multi_speaker.py \
  --text-file input_multiuser.txt \
  --output-file english_multi_speech.wav \
  --speaker-voices "SPEAKER0:../voice_prompts/en_woman.wav,../voice_prompts/en_woman.txt" \
  --speaker-voices "SPEAKER1:../voice_prompts/en_man.wav,../voice_prompts/en_man.txt" \
  --speed 1.2 \
  --temperature 0.4 \
  --max-tokens 2048
```

## 依赖安装

要使用速度控制功能，需要安装额外的依赖：

```bash
pip install librosa
```

如果没有安装librosa，脚本仍可正常运行，但速度控制功能将不可用，音频将以正常速度保存。

## 注意事项

1. 多说话人文本需要按照指定格式编写，每个说话人段落前需加上`[SPEAKER*]`标签
2. 每个说话人都可以指定不同的声音样本，实现个性化声音克隆
3. 速度控制功能需要安装librosa库（`pip install librosa`），否则音频将以正常速度保存
4. 速度质量选项允许在处理速度和音频质量之间进行权衡：
   - `high`：最高质量处理，但可能较慢
   - `medium`：平衡质量和处理速度
   - `low`：最快处理速度，但质量可能有所降低
5. 温度参数影响生成的随机性，较低值产生更确定的结果，较高值产生更多样化的结果
6. 如果模型路径未正确设置，脚本会自动使用HuggingFace Hub上的模型

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
3. 使用不同的声音样本为不同说话人