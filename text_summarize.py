import requests
from bs4 import BeautifulSoup
import ollama


class TextSummarizer:
    def __init__(self, model="llama3.2", max_length=4000):
        self.model = model
        self.max_length = max_length

    def extract_text_from_url(self, url):
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ", strip=True)

        return text

    def summarize_text(self, text):
        system_prompt = "You are an assistant that analyzes the contents of a website \
                        and provides a short summary, ignoring text that might be navigation related. \
                        Respond in markdown."

        user_prompt = f"The contents of this website is as follows; \
                        please provide a short summary of this website in markdown. \
                        If it includes news or announcements, then summarize these too.\n\n {text[:self.max_length]}"

        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]

    def summarize_url(self, url):
        text = self.extract_text_from_url(url)
        return self.summarize_text(text)


if __name__ == "__main__":
    url = input("Enter the URL to summarize: ")
    summarizer = TextSummarizer()
    try:
        summary = summarizer.summarize_url(url)
        print("\nSummary:\n", summary)
    except Exception as e:
        print("Error:", e)
