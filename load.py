from datasets import load_dataset

ds = load_dataset("tasksource/ruletaker")
ds.save_to_disk("data/ruletaker")
print(f"Датасет сохранён в data/ruletaker/")