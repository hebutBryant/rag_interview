from transformers import AutoTokenizer

from utils.base import print_text

#### Llama prompt template
# QA_preamble_llama_inst = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an intelligent AI assistant. Please answer questions based on the user's instructions. Below are some reference graph retrieval results that may help you in answering the user's question.\n\n"
# QA_query_llama_inst = "<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nPlease write a high-quantify answer for the given question using only the provided context information (some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\nQuestion: {question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>"

QA_SYSTEM = (
    "You are an intelligent AI assistant. Please answer questions based on the user's instructions. "
    "Below are some reference retrieval results that may help you in answering the user's question."
)

QA_USER = (
    "\nReference retrieval results:\n{context}\n\n"
    "Please write a high-quantify answer for the given question using only the provided context information "
    "(some of which might be irrelevant). Answer directly without explanation and keep the response short and direct.\n"
    "Question: {question}\n"
    "Answer: "
)


def build_prompt(model, system_prompt, user_prompt, **kwargs):

    tokenizer = AutoTokenizer.from_pretrained(model)

    messages = []

    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    try:
        messages.append({"role": "user", "content": user_prompt.format(**kwargs)})

    except KeyError as e:
        raise ValueError(f"Missing variable {e} in template: {user_prompt}")

    if getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "\n\n".join([msg["content"] for msg in messages])

    return prompt


if __name__ == "__main__":
    question = "What were the main objectives of the Apollo 11 mission?"

    context = (
        "Document 1:\n"
        "Apollo 11 was the first manned mission to land on the Moon. "
        "It was launched by NASA on July 16, 1969, with astronauts Neil Armstrong, "
        "Buzz Aldrin, and Michael Collins aboard.\n\n"
        "Document 2:\n"
        "The mission's main objectives included performing a crewed lunar landing, "
        "collecting samples from the Moon's surface, and returning safely to Earth.\n\n"
        "Document 3:\n"
        "On July 20, 1969, Neil Armstrong became the first human to step onto the lunar surface, "
        "followed by Buzz Aldrin. They collected 47.5 pounds of lunar material before returning on July 24."
    )

    model_list = [
        # "/home/hdd/model/Meta-Llama-3-8B",  # LLaMA3
        "/home/hdd/model/Meta-Llama-3-8B-Instruct",  # LLaMA3
        "/home/hdd/model/Mistral-7B-Instruct-v0.3",  # Mistral
        "/home/hdd/model/Qwen2-7B-Instruct",  # Qwen2
    ]

    for model in model_list:
        print(f"\n==== {model} ====")
        try:
            prompt_text = build_prompt(
                model, QA_SYSTEM, QA_USER, question=question, context=context
            )
            print_text(f"{prompt_text}", color="blue")
        except Exception as e:
            print(f"⚠️ Failed for {model}: {e}")
