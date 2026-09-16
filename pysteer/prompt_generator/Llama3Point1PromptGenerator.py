from prompt_generator.BasePromptGenerator import BasePromptGenerator


class Llama3Point1PromptGenerator(BasePromptGenerator):

    @staticmethod
    def _template_for_base_models(system: str, user: str) -> str:
        return f"<|begin_of_text|>{system}\n\n{user}\n\n"

    @staticmethod
    def _template_for_instruct_models(who_you_are: str, system: str, user: str) -> str:
        system_parts = []
        if who_you_are:
            system_parts.append(f"You are an expert in {who_you_are}.")
        if system:
            system_parts.append(system)
        system_content = " ".join(system_parts)
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    # The generate_prompt() method is inherited from BasePromptGenerator.
