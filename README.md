# phi3nedtuned-ner-json
- This model can be used to extract named entities such as item, quantity, price, total amount etc. from the receipts.
- This is an adaptor for base model microsoft/Phi-3-mini-4k-instruct and should be merged with it for deployment.

# To merge the adaptor with the base model:

```python

import torch
import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
​
#path where merged model will be saved
save_path = '.../phi3nedtuned-ner-json_merged'
​
#path for finetuned adaptor model
model_dir = ".../phi3nedtuned-ner-json"

config = PeftConfig.from_pretrained(model_dir)
​
#original phi-3 model

model = AutoModelForCausalLM.from_pretrained(
"microsoft/Phi-3-mini-4k-instruct",
device_map="cuda",
torch_dtype= 'auto',
trust_remote_code=True,
)
​
model = PeftModel.from_pretrained(model, model_dir)
model.config.to_json_file('adapter_config.json')

#merge
merged_model = model.merge_and_unload()
merged_model.save_pretrained(save_path)

```

# For inference, even without merge:

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

config = PeftConfig.from_pretrained("shujatoor/phi3nedtuned-ner-json")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)

model = PeftModel.from_pretrained(model, "shujatoor/phi3nedtuned-ner-json")
model.config.to_json_file('adapter_config.json')

torch.random.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained("shujatoor/phi3nedtuned-ner-json")


text = "Tehzeeb Bakers STRN3277876134234 Block A. Police Foundation,PwD Islamabad 051-5170713-4.051-5170501 STRN#3277876134234 NTN#7261076-2 Sales Receipt 05/04/202405:56:40PM CashierM J Payment:Cash Rate Qty. Total # Descriptlon 80.512.000 190.00 1.VEGETABLESAMOSA Sub Total 161.02 Total Tax: 28.98 POS Service Fee 1.00 Total 191.00 Cash 200.00 Change Due 9.00 SR#th007-220240405175640730 Goods Once Sold Can Not Be Taken Back or Replaced All Prices Are Inclusive Sales Tax 134084240405175640553"

q_json = "extracted_data': {'store_name': '', 'address': '', 'receipt_number': '', 'drug_license_number': '', 'gst_number': '', 'vat_number': '', 'date': '', 'time': '', 'items': [], 'total_items': '', 'gst_tax': '', 'vat_tax': '', 'gross_total': '', 'discount': '', 'net_total': '', 'contact': ''}"

qs = f'{text}. {q_json}'

print('Question:',qs, '\n')

messages = [
    #{"role": "system", "content": f" Only fill the provided JSON Object. Only output the answer, nothing else. Do not give output other than the  required JSON Object {q_json}"},
    {"role": "user", "content": qs},

]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 1024,
    "return_full_text": False,
    "temperature": 0.1,
    "do_sample": False,
}

output = pipe(messages, **generation_args)

print('Answer:', output[0]['generated_text'], '\n')
```
