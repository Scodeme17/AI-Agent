import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import requests
import threading
from typing import List, Dict
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from dotenv import load_dotenv
from pathlib import Path
import arxiv
import webbrowser
import os
import json

class ResearchGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Research Paper Finder")
        self.setup_styles()
        self.setup_gui()
        self.load_config()

    def setup_styles(self):
        style = ttk.Style()
        style.configure('Source.TButton',
                       padding=8,
                       font=('Helvetica', 10))
        style.configure('PDF.TButton',
                       padding=5,
                       font=('Helvetica', 9))

    def load_config(self):
        load_dotenv()
        self.IMAGE_DIR = Path("paper_images")
        self.IMAGE_DIR.mkdir(exist_ok=True)
        
        try:
            self.agent = Agent(model=OpenAIChat(model="gpt-4"))
            self.arxiv_client = arxiv.Client()
            self.google_api_key = os.getenv('GOOGLE_API_KEY')
            self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize services: {str(e)}")

    def google_search(self, query: str, num_results: int = 5) -> List[Dict]:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.google_api_key,
            'cx': self.google_cse_id,
            'q': query,
            'num': min(num_results, 10)
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"Google API error: {response.text}")
            
        data = response.json()
        return data.get('items', [])

    def setup_gui(self):
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky="nsew")

        input_frame = ttk.LabelFrame(main_container, text="Search Configuration", padding="10")
        input_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        ttk.Label(input_frame, text="Research Topic:", font=('Helvetica', 10, 'bold')).grid(row=0, column=0, padx=5)
        self.topic_entry = ttk.Entry(input_frame, width=50, font=('Helvetica', 10))
        self.topic_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(input_frame, text="Results per source:").grid(row=1, column=0, padx=5)
        self.num_results = ttk.Spinbox(input_frame, width=5)
        self.num_results.insert(0, "5")
        self.num_results.grid(row=1, column=1, sticky='w', padx=5)

        buttons_frame = ttk.Frame(input_frame)
        buttons_frame.grid(row=2, column=0, columnspan=2, pady=10)

        self.ieee_btn = ttk.Button(buttons_frame, text="IEEE Xplore", 
                                 style='Source.TButton',
                                 command=lambda: self.start_search('ieee'))
        self.ieee_btn.grid(row=0, column=0, padx=5)

        self.scholar_btn = ttk.Button(buttons_frame, text="Google Scholar",
                                    style='Source.TButton',
                                    command=lambda: self.start_search('scholar'))
        self.scholar_btn.grid(row=0, column=1, padx=5)

        self.others_btn = ttk.Button(buttons_frame, text="Others",
                                   style='Source.TButton',
                                   command=lambda: self.start_search('others'))
        self.others_btn.grid(row=0, column=2, padx=5)

        self.progress = ttk.Progressbar(input_frame, mode='indeterminate')
        self.progress.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)

        results_frame = ttk.LabelFrame(main_container, text="Results", padding="5")
        results_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.results_text = scrolledtext.ScrolledText(results_frame, height=30, width=100,
                                                    font=('Helvetica', 10))
        self.results_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

    def start_search(self, source: str):
        self.results_text.delete(1.0, tk.END)
        self.progress.start()
        threading.Thread(target=self._execute_search, args=(source,), daemon=True).start()

    def _execute_search(self, source: str):
        topic = self.topic_entry.get().strip()
        if not topic:
            self.show_error("Please enter a research topic")
            return

        try:
            num_results = int(self.num_results.get())
            if source == 'ieee':
                self.fetch_ieee_papers(topic, num_results)
            elif source == 'scholar':
                self.fetch_scholar_papers(topic, num_results)
            else:
                self.fetch_other_papers(topic, num_results)
        except Exception as e:
            self.show_error(f"Search error: {str(e)}")
        finally:
            self.root.after(0, self.progress.stop)

    def fetch_ieee_papers(self, topic: str, num_results: int):
        self.update_results("🔬 IEEE Xplore Papers:\n")
        try:
            results = self.google_search(
                f"{topic} site:ieeexplore.ieee.org", 
                num_results
            )
            for result in results:
                self.display_paper({
                    'title': result['title'],
                    'url': result['link'],
                    'snippet': result.get('snippet', ''),
                    'source': 'ieee'
                })
        except Exception as e:
            self.update_results(f"IEEE search error: {str(e)}\n")

    def fetch_scholar_papers(self, topic: str, num_results: int):
        self.update_results("🎓 Google Scholar Papers:\n")
        try:
            results = self.google_search(
                f"{topic} site:scholar.google.com", 
                num_results
            )
            for result in results:
                self.display_paper({
                    'title': result['title'],
                    'url': result['link'],
                    'snippet': result.get('snippet', ''),
                    'source': 'scholar'
                })
        except Exception as e:
            self.update_results(f"Scholar search error: {str(e)}\n")

    def fetch_other_papers(self, topic: str, num_results: int):
        self.update_results("📚 arXiv and Other Sources:\n")
        try:
            search = arxiv.Search(
                query=topic,
                max_results=num_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in self.arxiv_client.results(search):
                self.display_paper({
                    'title': paper.title,
                    'authors': [author.name for author in paper.authors],
                    'published': paper.published.strftime("%Y-%m-%d"),
                    'summary': paper.summary,
                    'pdf_url': paper.pdf_url,
                    'source': 'arxiv'
                })
        except Exception as e:
            self.update_results(f"ArXiv search error: {str(e)}\n")

    def display_paper(self, paper: Dict):
        if paper['source'] == 'arxiv':
            self.update_results(
                f"📑 Title: {paper['title']}\n"
                f"👥 Authors: {', '.join(paper['authors'])}\n"
                f"📅 Published: {paper['published']}\n"
                f"📝 Summary: {paper['summary'][:300]}...\n\n"
            )
            btn_text = "📥 Download PDF"
            url = paper['pdf_url']
        else:
            self.update_results(
                f"📑 Title: {paper['title']}\n"
                f"🔗 URL: {paper['url']}\n"
                f"📝 Preview: {paper['snippet']}\n\n"
            )
            btn_text = "🔍 View Paper"
            url = paper['url']

        view_button = ttk.Button(
            self.results_text,
            text=btn_text,
            style='PDF.TButton',
            command=lambda: webbrowser.open(url)
        )
        self.results_text.window_create(tk.END, window=view_button)
        self.update_results("\n\n")

    def update_results(self, text):
        self.root.after(0, lambda: self.results_text.insert(tk.END, text))
        self.root.after(0, lambda: self.results_text.see(tk.END))

    def show_error(self, message):
        self.root.after(0, lambda: messagebox.showerror("Error", message))
        self.root.after(0, self.progress.stop)

def main():
    root = tk.Tk()
    root.geometry("1000x800")
    app = ResearchGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()