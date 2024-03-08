fine-tune data: [fastchat-finetune/data/autodroid.json](fastchat-finetune/data/autodroid.json)

test data: [fastchat-finetune/data/questions.json](fastchat-finetune/data/questions.json)

## how to fine-tune:
- cd fastchat-finetune (Thanks to the official fastchat [repo](https://github.com/lm-sys/FastChat/))
- install according to the 'Install'
- bash finetune.sh (fine-tune the model)
- bash run_autodroid.sh (run the fine-tuned model on test data, which outputs the predicted answers in `autodroid_vicuna.json`)


## how to evaluate:
- cd evaluate
- python evaluate.py 
