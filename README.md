# 📚 WriteMeAStory

> A real-time AI story generator with streaming text generation and beautiful UI
> Deployed on Render: https://writemeastory.onrender.com

## ✨ Features

- 🎨 **Beautiful UI** - Modern gradient design with smooth animations
- ⚡ **Real-time Streaming** - Watch stories generate word-by-word using Socket.IO
- 🎛️ **Customizable Settings** - Adjust max tokens, temperature, and streaming options
- 📝 **Example Prompts** - Pre-built story starters to spark creativity
- 💾 **Download Stories** - Save generated stories as text files
- 🔄 **Auto-reconnection** - Robust WebSocket connection with fallback to HTTP
- 📱 **Responsive Design** - Works beautifully on desktop and mobile
- 🎯 **TypeScript** - Fully typed for better development experience

## 🏗️ Architecture

### Frontend Stack
- **React 18** with TypeScript
- **Socket.IO Client** for real-time communication
- **Tailwind CSS** for styling
- **Npm** for package managing
- **Lucide React** for icons
- **Custom Hooks** for WebSocket management

### Backend Stack

- **FastAPI** – Lightweight, high-performance Python web framework for building APIs.
- **Python-Socket.IO** – Enables real-time, bidirectional communication between client and server over WebSockets.
- **PyTorch** – Used to build and run the custom Transformer model for story generation.
- **Poetry** – Manages Python dependencies, environments, and project metadata.
- **Gunicorn + UvicornWorker** – Production-ready ASGI server setup for handling FastAPI with async support.
- **AsyncIO** – Powers non-blocking, token-level streaming during story generation.
- **CORS enabled** – Allows secure cross-origin requests from the React frontend during development.


### Project Structure
```
writeMeAStory/
├── frontend/
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   │   ├── Header.tsx
│   │   │   ├── PromptInput.tsx
│   │   │   ├── StoryOutput.tsx
│   │   │   └── ...
│   │   ├── hooks/               # Custom React hooks
│   │   │   └── useSocketIO.ts
│   │   ├── types/               # TypeScript definitions
│   │   │   └── index.ts
│   │   ├── Pages/
│   │   │   └── Home/
│   │   │       └── Home.tsx
│   │   └── App.tsx
│   ├── package.json
│   └── tailwind.config.js
├── backend/
│   ├── main.py                  # FastAPI + Socket.IO server
│   ├── poetry.lock
│   └── ...
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- **Node.js** 16+ and npm
- **Python** 3.8+ and pip

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Backend Setup
```bash
cd backend
poetry install
poetry run uvicorn main:app --reload
```

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/bdyldo/writeMeAStory.git
cd writeMeAStory
```

### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
```

### 3. Install Backend Dependencies
```bash
cd backend
poetry install
```
## 🎮 Usage

### Basic Story Generation
1. Enter a story prompt in the text area
2. Adjust settings (optional): max tokens, temperature, streaming
3. Click the send button or press `Ctrl/Cmd + Enter`
4. Watch your story generate in real-time!

### Advanced Features
- **Settings Panel**: Click the gear icon to customize generation parameters
- **Example Prompts**: Click any example to auto-fill the prompt
- **Download**: Save completed stories as `.txt` files
- **Connection Status**: Monitor real-time connection status

## 🔧 Configuration

### Frontend Settings
```typescript
// Default generation settings
const defaultSettings = {
  maxTokens: 200,        // Maximum story length
  temperature: 0.7,      // Creativity level (0-1)
  streamResponse: true   // Enable real-time streaming
};
```

### Backend Configuration
```python
# Socket.IO settings
sio = socketio.AsyncServer(
    cors_allowed_origins="*",
    async_mode='asgi'
)

# Generation parameters
max_tokens_limit = 500
temperature_range = (0.0, 1.0)
```

## 🌐 API Reference

### Socket.IO Events

#### Client → Server
```typescript
// Generate a new story
emit("generate_story", {
  prompt: string,
  max_tokens: number,
  temperature: number
})
```

#### Server → Client
```typescript
// Streaming story tokens
on("story_token", (data: { content: string }) => {})

// Generation complete
on("story_complete", () => {})

// Error handling
on("story_error", (data: { message: string }) => {})
```

### HTTP Endpoints
```
GET  /                    # Health check
POST /api/generate        # HTTP fallback for story generation
```

## 🛠️ Development

### Running in Development Mode
```bash
# Terminal 1 - Backend
cd backend
uvicorn main:socket_app --reload --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Building for Production
```bash
# Frontend
npm run build

# Backend
# Use production ASGI server like gunicorn
pip install gunicorn
gunicorn main:socket_app -w 4 -k uvicorn.workers.UvicornWorker
```

### Code Quality
```bash
# Frontend
npm run lint
npm run type-check

# Backend
pip install black flake8
black .
flake8 .
```

## 🐛 Troubleshooting

### Common Issues

**Stories Not Streaming**
```bash
# Verify Socket.IO connection
# Check network tab for WebSocket connection
# Ensure 'streamResponse' is enabled in settings
```

**Build Errors**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Check TypeScript errors
npm run type-check
```




