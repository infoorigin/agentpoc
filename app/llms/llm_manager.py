from llama_index.llms.openai import OpenAI

# Map model aliases to actual models (you can expand this)
model_configs = {
        "40-mini": {"model": "gpt-4o-mini"},  # example
        "4o": {"model": "gpt-4o"},
        "o1": {"model": "gpt-3.5-turbo-1106"},
        # Add more models if needed
    }

class LLMManager:
    _instances = {}

    @classmethod
    def get_llm(cls, model_name="4o"):
        """
        Get the LLM instance for a specific model alias.
        Lazily initialize if not already created.
        """
        if model_name not in cls._instances:
            if model_name not in model_configs:
                raise ValueError(f"Unknown model name '{model_name}' requested in LLMManager.")

            config = model_configs[model_name]
            print(f"🔵 Initializing LLM instance for model '{model_name}' ({config['model']})...")
            cls._instances[model_name] = OpenAI(
                model=config["model"]
            )

        return cls._instances[model_name]

    @classmethod
    def list_available_models(cls):
        return list(model_configs.keys())
