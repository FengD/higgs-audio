import base64
import os
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the audio file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def get_interleaved_dialogue_input_sample():
    system_prompt = (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: vocal fry;moderate pitch;monotone;masculine;young adult;slightly fast\n"
        "SPEAKER1: whispering;sad;trembling voice\n\n"
        "rain, distant thunder.\n"
        "<|scene_desc_end|>"
    )

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content="<|generation_instruction_start|>\nGenerate interleaved transcript and audio that lasts for around 20 seconds.\n<|generation_instruction_end|>",
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample


def get_zero_shot_input_sample():
    system_prompt = (
        "Generate audio following instruction.\n\n<|scene_desc_start|>\nSPEAKER0: crying accent\n<|scene_desc_end|>"
    )

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content="Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
            "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
            "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.",
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample


def get_voice_clone_input_sample():
    reference_text = "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
    reference_audio = encode_base64_content_from_file(
        os.path.join(os.path.dirname(__file__), "../voice_prompts/belinda.wav")
    )
    messages = [
        Message(
            role="user",
            content=reference_text,
        ),
        Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
        ),
        Message(
            role="user",
            content="我是浑元形意太极门掌门人马保国,刚才有个朋友问我:马老师发生什么事啦.我说怎么回事,给我发了几张截图,我一看,哦,原来是昨天,有两个年轻人,三十多岁,一个体重九十多公斤,一个体重八十多公斤.他们说,哎,有一个说是:我在健身房练功,颈椎练坏了,马老师你能不能教教我浑元功法",
        ),
    ]
    return ChatMLSample(messages=messages)


def get_voice_clone_input_sample_fr():
    reference_text = "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
    reference_audio = encode_base64_content_from_file(
        os.path.join(os.path.dirname(__file__), "../voice_prompts/belinda.wav")
    )
    messages = [
        Message(
            role="user",
            content=reference_text,
        ),
        Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
        ),
        Message(
            role="user",
            content="Le sexe est-il le problème fondamental que les hommes ont à résoudre ? C'est ce que pense Foucault, qui en 1976 fait paraître le 1er tome de l\"Histoire de la sexualité\". La sexualité que nous découvrons en nous nous appartient-elle ? Ou est-elle construite, comme tout mécanisme d'assujetissement ?",
        ),
    ]
    return ChatMLSample(messages=messages)


INPUT_SAMPLES = {
    "interleaved_dialogue": get_interleaved_dialogue_input_sample,
    "zero_shot": get_zero_shot_input_sample,
    "voice_clone": get_voice_clone_input_sample,
    "voice_clone_fr": get_voice_clone_input_sample_fr
}


def get_multi_speaker_input_sample():
    # Reference voices for two speakers
    reference_text_0 = "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year."
    reference_audio_0 = encode_base64_content_from_file(
        os.path.join(os.path.dirname(__file__), "../voice_prompts/belinda.wav")
    )
    
    reference_text_1 = "Welcome to the world of artificial intelligence. Today we'll explore the latest developments in neural networks."
    reference_audio_1 = encode_base64_content_from_file(
        os.path.join(os.path.dirname(__file__), "../voice_prompts/chadwick.wav")
    )
    
    messages = [
        # First speaker reference
        Message(
            role="user",
            content=f"[SPEAKER0] {reference_text_0}",
        ),
        Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio_0, audio_url="placeholder"),
        ),
        # Second speaker reference
        Message(
            role="user",
            content=f"[SPEAKER1] {reference_text_1}",
        ),
        Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio_1, audio_url="placeholder"),
        ),
        # Dialogue
        Message(
            role="user",
            content="[SPEAKER0] Hello there! I'm excited to discuss artificial intelligence with you today.",
        ),
        Message(
            role="user",
            content="[SPEAKER1] Likewise! I've been looking forward to this conversation. What aspects of AI interest you most?",
        ),
    ]
    return ChatMLSample(messages=messages)


INPUT_SAMPLES["multi_speaker"] = get_multi_speaker_input_sample

