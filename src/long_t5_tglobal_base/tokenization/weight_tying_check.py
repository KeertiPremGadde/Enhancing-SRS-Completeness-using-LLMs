from transformers import LongT5ForConditionalGeneration

model_path = "/home/gadde/Thesis/models/pretrained/long-t5-tglobal-base"
model = LongT5ForConditionalGeneration.from_pretrained(model_path)

print("Weight tying status:")
print("Tied?", model.lm_head.weight.data_ptr() == model.shared.weight.data_ptr())