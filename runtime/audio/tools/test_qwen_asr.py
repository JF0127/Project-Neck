from qwen_asr import Qwen3ASRModel

MODEL_PATH = "/home/jhl/projects/Project-Neck/dataset/models/qwen"

AUDIO_PATH = (
    "/home/jhl/projects/Project-Neck/dataset/datasets/"
    "zhubo_shuo_lianbo/clean_v2/male/kanghui/"
    "n0Zvshs8wzk_a0421f2b63/"
    "n0Zvshs8wzk_a0421f2b63.wav"
)

asr = Qwen3ASRModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype="bfloat16",
)

result = asr.transcribe(AUDIO_PATH)

print(result)
