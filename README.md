# Coding Assistant 

A sophisticated desktop application combining AI-powered coding assistance with an integrated multi-language compiler, designed to enhance developer productivity.

## Features

### AI-Powered Coding Assistant
- **Smart Query Handling**: Recognizes 500+ programming-related keywords across multiple domains
- **Code Block Extraction**: Automatically detects and executes code snippets from responses
- **Contextual Memory**: Maintains chat history and suggests similar previous questions
- **Groq Integration**: Utilizes the Groq API with LLaMA 3 70B model for high-quality responses

### Integrated Development Environment
- **Multi-language Support**: 
  - Python, Java, C, C++, JavaScript
- **Real-time Execution**:
  - Syntax validation
  - Code execution with timeout protection
  - Output/error display
- **Code Management**:
  - Auto-formatting (Python via autopep8)
  - Versioned saving with timestamp
  - Direct execution from chat responses

### User Experience
- **Customizable UI**:
  - Light/Dark theme switching
  - Adjustable font size (8-20pt)
  - Responsive paned interface
- **History Management**:
  - Persistent chat history
  - Configurable similar questions memory
  - Auto-save functionality

## Installation
pip install phi python-dotenv 
### Requirements
- Python 3.8+
- Tkinter (usually included with Python)
- Groq API key (free tier available)
