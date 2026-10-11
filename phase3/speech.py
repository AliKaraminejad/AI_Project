import queue
import threading


class SpeechWorker:
    def __init__(self, enabled: bool = True, rate: int = 165) -> None:
        self.enabled = enabled
        self.rate = rate
        self.error: str | None = None
        self.items: queue.Queue[str | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        if not self.enabled:
            return
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            while True:
                text = self.items.get()
                if text is None:
                    break
                engine.say(text)
                engine.runAndWait()
        except Exception as error:
            self.error = str(error)

    def speak(self, text: str) -> None:
        value = text.strip()
        if self.enabled and value:
            self.items.put(value)

    def close(self) -> None:
        if self.thread.is_alive():
            self.items.put(None)
            self.thread.join(timeout=2)
