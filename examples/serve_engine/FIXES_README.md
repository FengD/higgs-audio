# TTS API 音频播放问题修复说明

## 问题描述
TTS服务生成音频后，Web端无法打开或播放音频文件。

## 问题原因分析
1. 前端页面中的音频路径引用不正确
2. 后端API返回的文件路径格式与前端期望的不匹配
3. 缺少正确的静态文件服务配置

## 修复方案

### 1. 后端API修复 (`tts_api_service.py`)
- 修改API响应，返回相对于服务器的文件路径而不是绝对路径
- 添加日志记录以便调试
- 增加了两个音频文件服务端点：
  - `/tmp/{file_path}` - 直接从/tmp目录提供文件
  - `/audio/{file_path}` - 原有的音频文件服务端点

### 2. 前端页面修复 (`templates/index.html`)
- 更新音频文件路径处理逻辑，确保能正确引用生成的音频文件
- 简化了音频源的路径构造

## 如何测试修复

1. 启动TTS API服务（需要先激活conda环境）：
   ```bash
   conda activate sglang
   cd examples/serve_engine
   python start_tts_api.py
   ```

2. 打开浏览器访问 http://localhost:8000

3. 输入多说话人文本，设置说话人映射，点击"生成语音"

4. 生成完成后，应该能在结果区域直接播放音频

## 或者使用测试脚本验证

运行测试脚本来验证修复是否有效：
```bash
python test_fix.py
```

## 技术细节

### API端点变更
- **POST** `/tts/multi-speaker` - 现在返回相对于服务器的文件路径
- **GET** `/tmp/{file_path}` - 新增端点，直接从/tmp目录提供音频文件
- **GET** `/audio/{file_path}` - 原有端点保持兼容性

### 响应格式示例
```json
{
  "message": "Successfully generated speech",
  "output_file": "/tmp/tts_output_xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.wav",
  "speakers_detected": ["SPEAKER0", "SPEAKER1"]
}
```

## 注意事项
1. 确保/tmp目录有适当的读取权限
2. 生成的音频文件会在请求处理完成后保存在/tmp目录中
3. 浏览器需要能够访问API服务器提供的静态文件