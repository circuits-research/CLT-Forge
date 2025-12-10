import safetensors
data = safetensors.safe_open("/home/abir19/scratch/data/featflow/activations_gpt2_multilingual_20/ctx_16/activations_split_0.safetensors", framework="pt")
print({k: data.get_tensor(k).shape for k in data.keys()})