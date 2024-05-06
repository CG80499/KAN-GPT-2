from datasets import load_dataset
import numpy as np
import tiktoken

EOS_TOKEN = "<|endoftext|>"

TOKENIZER_SIZE = 50257

class TinyStoriesDataset:

    def __init__(self, *, max_len: int):
        print("Loading dataset...")
        dataset = load_dataset("roneneldan/TinyStories")
        self.texts = [sample["text"]+EOS_TOKEN for sample in dataset["train"]]
        self.enc = tiktoken.encoding_for_model("gpt2")
        print("Dataset loaded.")
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def create_batches(self, batch_size):
        current_index = 0
        while current_index + batch_size < len(self.texts):
            batch_texts = self.texts[current_index:current_index+batch_size]
            batch_encodings = self.enc.encode_batch(batch_texts, allowed_special={EOS_TOKEN})
            batch_encodings = [encoding[:self.max_len] for encoding in batch_encodings]
            batch_encodings_sizes = [len(encoding) for encoding in batch_encodings]
            # create masks, they should 1 where there is a token and 0 where there is padding
            batch_masks = np.zeros((batch_size, self.max_len), dtype=np.float32)
            for i, size in enumerate(batch_encodings_sizes):
                batch_masks[i, :size] = 1
            batch_encodings = [encoding + [0] * (self.max_len - len(encoding)) for encoding in batch_encodings]
            yield np.array(batch_encodings), batch_masks
            current_index += batch_size


# dataset = TinyStoriesDataset(max_len=256)

# for batch, mask in dataset.create_batches(2):
#     print(batch)
#     print(mask)
#     break

# print("Datset size:", len(dataset))