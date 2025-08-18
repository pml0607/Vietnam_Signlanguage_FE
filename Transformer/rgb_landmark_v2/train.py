import fire
from pathlib import Path
#add parrent dir to easy import
import sys
import os
sys.path.append(os.path.dirname(__file__))
#model 
from Model.model import VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
#dataloader 
from dataset import get_dataset, collate_fn
from dataset.utils import get_label_map

#metric funtion
from _fn.metric import compute_metrics

def main(dataset_root_path= "/work/21013187/SAM-SLR-v2/data/rgb_filtered",
         model_ckpt = "MCG-NJU/videomae-base" 
         ,num_epochs: int = 30,
         batch_size: int = 20):
    
    #prepare dataset info
    dataset_root_path = Path(dataset_root_path)
    label2id, id2label = get_label_map(dataset_root_path)
    #prepare model
    # the last classify layer with change num class output with len(label2id)
    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True
    )

    train_dataset, val_dataset, test_dataset = get_dataset(dataset_root_path)
    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-finetuned_47classes_raw"
    print(f"Batch size: {batch_size}") 
    print(f"Epochs : {num_epochs}")
    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-6,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        report_to="none",
        ddp_find_unused_parameters=False  # Set to True if you have unused parameters
    )
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        
    )
    
    train_results = trainer.train()
    trainer.evaluate(test_dataset)
    
    trainer.save_model()
    test_results = trainer.evaluate(test_dataset)
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    trainer.save_state()

if __name__ == '__main__':
    fire.Fire(main)