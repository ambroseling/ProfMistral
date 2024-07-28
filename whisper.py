from tqdm import tqdm
import whisperx
import gc 
import os
import jsonlines

device = "cuda" 
audio_file = "/home/tiny_ling/projects/my_mistral/audios"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
with jsonlines.open('data.jsonl', mode='w') as writer:
    for i,file in enumerate(tqdm(os.listdir(audio_file))):
        audio = whisperx.load_audio(os.path.join(audio_file,file))
        result = model.transcribe(audio, batch_size=batch_size)
        for obj in result['segments']:
            obj.pop('start')
            obj.pop('end')
            writer.write(obj)