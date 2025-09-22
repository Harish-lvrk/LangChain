from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "sshleifer/distilbart-cnn-12-6"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0)
text = """A Hare was always boasting about how fast he was, while the Tortoise was known for being slow and steady. One day, the Hare made fun of the Tortoise for being so slow. The Tortoise, tired of the Hare's teasing, challenged him to a race, which the Hare readily accepted. 
The race began, and the Hare quickly shot off like a rocket, leaving the Tortoise far behind. After running for a while, the Hare looked back and saw the Tortoise was nowhere in sight. Confident that he was so far ahead, the Hare decided to take a nap under a tree, thinking he had plenty of time. 
While the Hare slept soundly, the slow and steady Tortoise kept moving, never stopping. He slowly but surely passed the sleeping Hare. When the Tortoise finally reached the finish line, he had won the race. The Hare woke up and, seeing the Tortoise had crossed the finish line, was shocked to realize he had lost. 
"""
result = summarizer(text, max_length=130, min_length=30, do_sample=False)
print(result[0]['summary_text'])
