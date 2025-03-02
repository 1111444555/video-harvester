from abc import ABC, abstractmethod
import torch
from threading import Lock


class Translator(ABC):
    _instances = {}
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """
        Ensure that only one instance of each subclass is created (Singleton pattern).
        """
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super(Translator, cls).__new__(cls)
        return cls._instances[cls]

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_model(self, device: str) -> None:
        """
        Move the model to the specified device (CPU/GPU).
        """
        self.device = torch.device(device)
        if hasattr(self, "model"):
            self.model.to(self.device)

    @abstractmethod
    def translate(self, input_text: str, src_lang: str, dst_lang: str) -> str:
        """
        Translates content from input to output.
        """
        pass


class MbartTranslator(Translator):
    def __init__(self) -> None:
        """
        Load the MBART model and tokenizer.
        """
        super().__init__()
        from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

        if not hasattr(self, "model"):
            self.model = MBartForConditionalGeneration.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt"
            )
            self.tokenizer = MBart50TokenizerFast.from_pretrained(
                "facebook/mbart-large-50-many-to-many-mmt"
            )
            self.to_model(self.device.type)

    def translate(self, input_text: str, src_lang: str, dst_lang: str) -> str:
        """
        Translate from src_lang to dst_lang using MBART model.
        """
        self.tokenizer.src_lang = src_lang
        encoded_tokens = self.tokenizer(input_text, return_tensors="pt")
        encoded_tokens = encoded_tokens.to(self.device)

        generated_tokens = self.model.generate(
            **encoded_tokens,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[dst_lang],
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
