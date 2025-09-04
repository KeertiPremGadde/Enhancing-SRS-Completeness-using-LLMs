#inference for non PEFT models
import os
import json
import torch
import re
from transformers import LEDTokenizer, LEDForConditionalGeneration
from datetime import datetime
import sys
sys.path.append("/home/gadde/Thesis/src/led-base-16384")  # Ensure module visibility
#from preprocessing.data_loading0 import LEDDataLoader  # Uses your existing flatten_srs_to_text
#from preprocessing.data_loading_revival import flatten_srs_to_text
from preprocessing.data_loading_revival import LEDDataLoader
import logging

# ------------------- CONFIG -------------------
MODEL_DIR = "/home/gadde/Thesis/experiments/led-base-16384/2025_08_02-00:51:04_lr5e-06_bs1_epochs10_run001/checkpoints"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")
TOKENIZER_PATH = "/home/gadde/Thesis/models/pretrained/led-base-16384-updated"

INPUT_JSON = "/home/gadde/Thesis/data/infer/incomplete_srs_00219.json"
OUTPUT_DIR = "/home/gadde/Thesis/src/led-base-16384/inference/2025_08_02-00:51:04_lr5e-06_bs1_epochs10_run001/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_JSON = os.path.join(
    OUTPUT_DIR, "complete_srs_00219.json"
)

# Optional limits (must match training config)
MAX_INPUT_LEN = 925
MAX_OUTPUT_LEN = 1024

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LED_Inference")

# ------------------- INFERENCE -------------------
def load_model_and_tokenizer():
    logger.info("Loading tokenizer and model...")
    tokenizer = LEDTokenizer.from_pretrained(TOKENIZER_PATH)
    print(tokenizer.convert_tokens_to_ids('[SEC]'))
    print(tokenizer.convert_tokens_to_ids('[SUBSEC]'))
    print(tokenizer.convert_tokens_to_ids('[SUBSUBSEC]'))
    #tokenizer.add_tokens(['[SEC]', '[SUBSEC]', '[SUBSUBSEC]'])

    model = LEDForConditionalGeneration.from_pretrained(TOKENIZER_PATH)
    #model.resize_token_embeddings(len(tokenizer))

    checkpoint = torch.load(MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer

def prepare_input(input_json, tokenizer):
    with open(input_json, 'r') as f:
        srs_data = json.load(f)

    flat_text = LEDDataLoader.flatten_srs_to_text(srs_data)  # Using static-like call
    #flat_text = flatten_srs_to_text(None, srs_data)

    inputs = tokenizer(
        flat_text,
        max_length=MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    global_attention_mask = torch.zeros_like(inputs['input_ids'])
    for i, token_id in enumerate(inputs['input_ids'][0]):
        token = tokenizer.decode([token_id])
        if any(marker in token for marker in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']):
            global_attention_mask[0][i] = 1

    return inputs, global_attention_mask

def generate_output(model, tokenizer, inputs, global_attention_mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    global_attention_mask = global_attention_mask.to(device)

    #decoder_start_token_id = tokenizer.convert_tokens_to_ids('[SEC]')
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            global_attention_mask=global_attention_mask,
            #decoder_start_token_id=decoder_start_token_id,
            max_length=MAX_OUTPUT_LEN,
            num_beams=4,
            early_stopping=False,
            #no_repeat_ngram_size=4,          # Avoid repetition
            length_penalty=0.9,              # Slight encouragement to keep generating
            
            do_sample=True,                 # Try sampling
            top_p=0.92,                     # Nucleus sampling
            top_k=20                       # Prevent extreme randomness
            #no_repeat_ngram_size=20         # Allow structure patterns to repeat a bit
        )

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    decoded_raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]


    print("Generated token count:", generated_ids.shape[1])
    print("Decoded output length (chars):", len(decoded_raw))
    print("Input token length:", inputs['input_ids'].shape[1])
    
    print("\U0001f9fe Decoded output:\n", decoded)  # ⬅️ Debug print
    print("\U0001f9fe Decoded raw output:\n", decoded_raw)  # ⬅️ Debug print
    # Log both
    logger.info(f" Decoded (clean):\n{decoded}")
    logger.info(f" Decoded (raw):\n{decoded_raw}")
    
    # check
    if not any(tok in decoded_raw for tok in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']):
        logger.warning("⚠️ Missing structure tokens in decoded raw output!")
    return decoded, decoded_raw

# ✅ INSERT HERE
def preprocess_output_for_parsing(output):
    output = re.sub(r"(\[SEC\])", r"\n\1", output)
    output = re.sub(r"(\[SUBSEC\])", r"\n\t\1", output)
    output = re.sub(r"(\[SUBSUBSEC\])", r"\n\t\t\1", output)
    return output.strip()




# def save_output_as_json(text_output, output_path):
#     structured = parse_srs_to_json(text_output)
#     print("🔍 Parsed sections:", structured.keys())  # Optional debug
#     with open(output_path, 'w') as f:
#         json.dump(structured, f, indent=2)
#     logger.info(f"✓ Structured JSON saved at: {output_path}")




#New
def save_output_as_json(text_output, output_path):
    preprocessed = preprocess_output_for_parsing(text_output)  # 👈 Preprocess structure tokens
    structured = parse_srs_to_json(preprocessed)
    print("🔍 Parsed sections:", structured.keys())  # Optional debug
    with open(output_path, 'w') as f:
        json.dump(structured, f, indent=2)
    logger.info(f"✓ Structured JSON saved at: {output_path}")

    # ✅ Step 4: Optional — save the readable preprocessed output to a .txt file
    txt_path = output_path.replace(".json", ".txt")
    with open(txt_path, "w") as f:
        f.write(preprocessed)
    logger.info(f" Raw preprocessed output saved at: {txt_path}")



def parse_srs_to_json(text):
     # ⚠️ Check for structure tokens
    missing = []
    if '[SEC]' not in text:
        missing.append('[SEC]')
    if '[SUBSEC]' not in text:
        missing.append('[SUBSEC]')
    if '[SUBSUBSEC]' not in text:
        missing.append('[SUBSUBSEC]')

    if missing:
        logger.warning(f" Missing structure tokens in output: {', '.join(missing)}. Output may lack hierarchy.")
    
    lines = text.strip().splitlines()
    structured = {}
    current_sec, current_subsec, current_subsubsec = None, None, None
    buffer = ""
    last_tag = (None, None, None)

    def insert(section, subsection=None, subsubsection=None, content=""):
        if subsection is None:
            structured[section] = content
        elif subsubsection is None:
            structured.setdefault(section, {})[subsection] = content
        else:
            structured.setdefault(section, {}).setdefault(subsection, {})[subsubsection] = content

    for line in lines:
        match = re.match(r"(\[SEC\]|\[SUBSEC\]|\[SUBSUBSEC\]) ([0-9.]+) (.+)", line.strip())
        if match:
            if buffer.strip():
                insert(*last_tag, buffer.strip())

            tag, number, title = match.groups()
            full_title = f"{tag} {number} {title}"

            if tag == "[SEC]":
                current_sec = full_title
                current_subsec, current_subsubsec = None, None
                last_tag = (current_sec, None, None)
            elif tag == "[SUBSEC]":
                current_subsec = full_title
                current_subsubsec = None
                last_tag = (current_sec, current_subsec, None)
            elif tag == "[SUBSUBSEC]":
                current_subsubsec = full_title
                last_tag = (current_sec, current_subsec, current_subsubsec)

            buffer = ""
        else:
            buffer += line.strip() + " "

    if buffer.strip():
        insert(*last_tag, buffer.strip())

    return structured

# ------------------- MAIN -------------------
def main():
    logger.info("Starting inference...")
    model, tokenizer = load_model_and_tokenizer()
    inputs, global_attention_mask = prepare_input(INPUT_JSON, tokenizer)
    output_clean, output_raw = generate_output(model, tokenizer, inputs, global_attention_mask)
    save_output_as_json(output_raw, OUTPUT_JSON)

if __name__ == "__main__":
    main()























































# #inference for PEFT models
# from peft import PeftModel, PeftConfig
# from transformers import LEDTokenizer
# from transformers import LEDForConditionalGeneration
# import os
# import json
# import torch
# import re
# from transformers import LEDTokenizer, LEDForConditionalGeneration
# from datetime import datetime
# import sys
# sys.path.append("/home/gadde/Thesis/src/led-base-16384")  # ✨ Ensure module visibility
# from preprocessing.data_loading import LEDDataLoader  # Uses your existing flatten_srs_to_text
# import logging

# # ------------------- CONFIG -------------------
# MODEL_DIR = "/home/gadde/Thesis/experiments/led-base-16384/2025_04_02-15:37:57_lr5e-06_bs1_epochs20_run001/checkpoints"
# MODEL_PATH = os.path.join(MODEL_DIR, "epoch_10.pth")
# TOKENIZER_PATH = "/home/gadde/Thesis/models/pretrained/led-base-16384-updated"

# INPUT_JSON = "/home/gadde/Thesis/data/test/easy/requirement_coverage/incomplete_srs_00211_00002.json"
# OUTPUT_DIR = "/home/gadde/Thesis/src/led-base-16384/inference/2025_04_02-15:37:57_lr5e-06_bs1_epochs20_run001"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# OUTPUT_JSON = os.path.join(
#     OUTPUT_DIR, "inference_complete_srs_00211_00002_3rdApril.json"
# )

# # Optional limits (must match training config)
# MAX_INPUT_LEN = 925
# MAX_OUTPUT_LEN = 1024

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("LED_Inference")

# # ------------------- INFERENCE -------------------
# def load_model_and_tokenizer():
#     # logger.info("Loading tokenizer and LoRA PEFT model...")

#     # # Load tokenizer from the same directory where you saved tokenizer files during training
#     # tokenizer = LEDTokenizer.from_pretrained(MODEL_DIR)
#     # #tokenizer.add_tokens(['[SEC]', '[SUBSEC]', '[SUBSUBSEC]'])

#     # # Load base model from the config inside adapter
#     # peft_config = PeftConfig.from_pretrained(MODEL_DIR)
#     # base_model = LEDForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)

#     # # Load LoRA adapter weights into the base model
#     # model = PeftModel.from_pretrained(base_model, MODEL_DIR)
#     # model.resize_token_embeddings(len(tokenizer))  # Resize if you added special tokens

#     # model.eval()
#     # return model, tokenizer

#     logger.info("Loading tokenizer and LoRA PEFT model...")

#     # Load tokenizer (vocab + special tokens already present)
#     tokenizer = LEDTokenizer.from_pretrained(MODEL_DIR)
#     print("🧠 Tokenizer length:", len(tokenizer))
#     print("🧠 Token IDs:", tokenizer.convert_tokens_to_ids(['[SEC]', '[SUBSEC]', '[SUBSUBSEC]', '[/SEC]']))
#     print("🪪 eos_token_id:", tokenizer.eos_token_id, tokenizer.decode(tokenizer.eos_token_id))

#     # Load base model config from adapter
#     peft_config = PeftConfig.from_pretrained(MODEL_DIR)
#     base_model = LEDForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)

#     # Apply LoRA adapter
#     model = PeftModel.from_pretrained(base_model, MODEL_DIR)

#     # Resize to match tokenizer
#     #model.resize_token_embeddings(len(tokenizer))
#     print("🧠 Model embedding size:", model.get_input_embeddings().weight.size(0))

#     model.eval()
#     return model, tokenizer

# def prepare_input(input_json, tokenizer):
#     with open(input_json, 'r') as f:
#         srs_data = json.load(f)

#     flat_text = LEDDataLoader.flatten_srs_to_text(None, srs_data)  # Using static-like call

#     inputs = tokenizer(
#         flat_text,
#         max_length=MAX_INPUT_LEN,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     )

#     global_attention_mask = torch.zeros_like(inputs['input_ids'])
#     for i, token_id in enumerate(inputs['input_ids'][0]):
#         token = tokenizer.decode([token_id])
#         if any(marker in token for marker in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']):
#             global_attention_mask[0][i] = 1

#     return inputs, global_attention_mask

# def generate_output(model, tokenizer, inputs, global_attention_mask):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     global_attention_mask = global_attention_mask.to(device)

#     #bad_words_ids = [[tokenizer.eos_token_id]]  # i.e., [[2]]
#     decoder_start_token_id = tokenizer.convert_tokens_to_ids('[SEC]')
#     with torch.no_grad():
#         generated_ids = model.generate(
#             # input_ids=inputs['input_ids'],
#             # attention_mask=inputs['attention_mask'],
#             # global_attention_mask=global_attention_mask,
#             # decoder_start_token_id=decoder_start_token_id,
#             # bad_words_ids=bad_words_ids,
#             # max_length=MAX_OUTPUT_LEN,
#             # num_beams=4,
#             # no_repeat_ngram_size=4,          # ⬅️ Avoid repetition
#             # length_penalty=0.9,              # ⬅️ Slight encouragement to keep generating
            
#             # do_sample=True,                 # ⬅️ Try sampling
#             # top_p=0.92,                     # ⬅️ Nucleus sampling
#             # top_k=20,                       # ⬅️ Prevent extreme randomness
#             # #no_repeat_ngram_size=20         # ⬆️ Allow structure patterns to repeat a bit
#             # early_stopping=False,                # ⬅️ This disables stopping on EOS
#             # eos_token_id=None,
#             # repetition_penalty=1.0  # No penalty
#             # #no_repeat_ngram_size=0  # Allow repeating structure tokens                    # ⬅️ Prevents stopping on </s>


#             input_ids=inputs['input_ids'],
#             attention_mask=inputs['attention_mask'],
#             global_attention_mask=global_attention_mask,
#             decoder_start_token_id=tokenizer.convert_tokens_to_ids('[SEC]'),
#             max_length=1024,
#             num_beams=4,
#             do_sample=True,
#             top_p=0.92,
#             top_k=20,
#             length_penalty=0.9,
#             eos_token_id=None,
#             bad_words_ids=[[tokenizer.eos_token_id]],
#             no_repeat_ngram_size=0,
#             repetition_penalty=1.1,
#             early_stopping=False
#         )

#     decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
#     decoded_raw = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

#     # 🔍 Add this
#     print("Generated token count:", generated_ids.shape[1])
#     print("Decoded output length (chars):", len(decoded_raw))
#     print("Input token length:", inputs['input_ids'].shape[1])
    
#     print("\U0001f9fe Decoded output:\n", decoded)  # ⬅️ Debug print
#     print("\U0001f9fe Decoded raw output:\n", decoded_raw)  # ⬅️ Debug print
#     # Log both
#     logger.info(f"🧾 Decoded (clean):\n{decoded}")
#     logger.info(f"🧾 Decoded (raw):\n{decoded_raw}")
#     # 👇 Add this check here
#     if not any(tok in decoded_raw for tok in ['[SEC]', '[SUBSEC]', '[SUBSUBSEC]']):
#         logger.warning("⚠️ Missing structure tokens in decoded raw output!")
#     return decoded, decoded_raw

# def save_output_as_json(text_output, output_path):
#     structured = parse_srs_to_json(text_output)
#     print("🔍 Parsed sections:", structured.keys())  # Optional debug
#     with open(output_path, 'w') as f:
#         json.dump(structured, f, indent=2)
#     logger.info(f"✓ Structured JSON saved at: {output_path}")

# def parse_srs_to_json(text):
#      # ⚠️ Check for structure tokens
#     missing = []
#     if '[SEC]' not in text:
#         missing.append('[SEC]')
#     if '[SUBSEC]' not in text:
#         missing.append('[SUBSEC]')
#     if '[SUBSUBSEC]' not in text:
#         missing.append('[SUBSUBSEC]')

#     if missing:
#         logger.warning(f"⚠️ Missing structure tokens in output: {', '.join(missing)}. Output may lack hierarchy.")
    
#     lines = text.strip().splitlines()
#     structured = {}
#     current_sec, current_subsec, current_subsubsec = None, None, None
#     buffer = ""
#     last_tag = (None, None, None)

#     def insert(section, subsection=None, subsubsection=None, content=""):
#         if subsection is None:
#             structured[section] = content
#         elif subsubsection is None:
#             structured.setdefault(section, {})[subsection] = content
#         else:
#             structured.setdefault(section, {}).setdefault(subsection, {})[subsubsection] = content

#     for line in lines:
#         match = re.match(r"(\[SEC\]|\[SUBSEC\]|\[SUBSUBSEC\]) ([0-9.]+) (.+)", line.strip())
#         if match:
#             if buffer.strip():
#                 insert(*last_tag, buffer.strip())

#             tag, number, title = match.groups()
#             full_title = f"{tag} {number} {title}"

#             if tag == "[SEC]":
#                 current_sec = full_title
#                 current_subsec, current_subsubsec = None, None
#                 last_tag = (current_sec, None, None)
#             elif tag == "[SUBSEC]":
#                 current_subsec = full_title
#                 current_subsubsec = None
#                 last_tag = (current_sec, current_subsec, None)
#             elif tag == "[SUBSUBSEC]":
#                 current_subsubsec = full_title
#                 last_tag = (current_sec, current_subsec, current_subsubsec)

#             buffer = ""
#         else:
#             buffer += line.strip() + " "

#     if buffer.strip():
#         insert(*last_tag, buffer.strip())

#     return structured

# # ------------------- MAIN -------------------
# def main():
#     logger.info("Starting inference...")
#     model, tokenizer = load_model_and_tokenizer()
#     inputs, global_attention_mask = prepare_input(INPUT_JSON, tokenizer)
#     output_clean, output_raw = generate_output(model, tokenizer, inputs, global_attention_mask)
#     save_output_as_json(output_raw, OUTPUT_JSON)

# if __name__ == "__main__":
#     main()