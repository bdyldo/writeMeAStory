import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Loader,
  BookOpen,
  Sparkles,
  Settings,
  Download,
} from "lucide-react";

const StoryGenerator = () => {
  const [prompt, setPrompt] = useState("");
  const [story, setStory] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [settings, setSettings] = useState({
    maxTokens: 200,
    temperature: 0.7,
    streamResponse: true,
  });

  const [showSettings, setShowSettings] = useState<boolean>(false);
  const [connectionStatus, setConnectionStatus] = useState("Disconnected");

  const storyRef = useRef<HTMLDivElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // WebSocket connection for streaming
  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    try {
      // In production, replace with your backend URL
      const wsUrl = "ws://localhost:8000/ws";
      wsRef.current = new WebSocket(wsUrl);

      wsRef.current.onopen = () => {
        setConnectionStatus("connected");
        console.log("WebSocket connected");
      };

      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "token") {
          setStory((prev) => prev + data.content);
        } else if (data.type === "complete") {
          setIsGenerating(false);
        } else if (data.type === "error") {
          console.error("Generation error:", data.content);
          setIsGenerating(false);
        }
      };

      wsRef.current.onclose = () => {
        setConnectionStatus("Disconnected");
        setTimeout(connectWebSocket, 3000); // Auto-reconnect
      };

      wsRef.current.onerror = (error) => {
        console.error("WebSocket error:", error);
        setConnectionStatus("error");
      };
    } catch (error) {
      console.error("Failed to connect WebSocket:", error);
      setConnectionStatus("error");
    }
  };

  // Auto-scroll to bottom of story
  useEffect(() => {
    if (storyRef.current) {
      storyRef.current.scrollTop = storyRef.current.scrollHeight;
    }
  }, [story]);

  const generateStory = async () => {
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);
    setStory("");

    if (
      settings.streamResponse &&
      wsRef.current?.readyState === WebSocket.OPEN
    ) {
      // Use WebSocket for streaming
      const request = {
        type: "generate",
        prompt: prompt.trim(),
        max_tokens: settings.maxTokens,
        temperature: settings.temperature,
      };
      wsRef.current.send(JSON.stringify(request));
    } else {
      // Fallback to HTTP API
      try {
        const response = await fetch("http://localhost:8000/generate", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt: prompt.trim(),
            max_tokens: settings.maxTokens,
            temperature: settings.temperature,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setStory(data.generated_text);
      } catch (error) {
        console.error("Generation failed:", error);
        setStory(
          "Error: Failed to generate story. Please check your connection."
        );
      } finally {
        setIsGenerating(false);
      }
    }
  };

  const handleKeyPress = (e:) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      generateStory();
    }
  };

  const downloadStory = () => {
    if (!story) return;

    const element = document.createElement("a");
    const file = new Blob([`Prompt: ${prompt}\n\n${story}`], {
      type: "text/plain",
    });
    element.href = URL.createObjectURL(file);
    element.download = `story_${Date.now()}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const examplePrompts = [
    "Once upon a time in a distant galaxy,",
    "The old lighthouse keeper discovered something unusual in the fog that night:",
    "Detective Sarah Chen had seen many strange cases, but this one was different:",
    "In the year 2157, humanity's first time traveler returned with urgent news:",
    "The ancient book glowed softly as Maya opened it for the first time:",
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      {/* Header */}
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-3">
            <BookOpen className="h-8 w-8 text-purple-400" />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              AI Story Generator
            </h1>
          </div>

          <div className="flex items-center space-x-4">
            <div
              className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
                connectionStatus === "connected"
                  ? "bg-green-500/20 text-green-400"
                  : connectionStatus === "error"
                  ? "bg-red-500/20 text-red-400"
                  : "bg-yellow-500/20 text-yellow-400"
              }`}
            >
              <div
                className={`w-2 h-2 rounded-full ${
                  connectionStatus === "connected"
                    ? "bg-green-400"
                    : connectionStatus === "error"
                    ? "bg-red-400"
                    : "bg-yellow-400"
                }`}
              />
              {connectionStatus}
            </div>

            <button
              onClick={() => setShowSettings(!showSettings)}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
            >
              <Settings className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Settings Panel */}
        {showSettings && (
          <div className="mb-8 p-6 bg-white/10 rounded-xl backdrop-blur-sm">
            <h3 className="text-lg font-semibold mb-4">Generation Settings</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">
                  Max Tokens
                </label>
                <input
                  type="number"
                  min="50"
                  max="500"
                  value={settings.maxTokens}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      maxTokens: parseInt(e.target.value),
                    }))
                  }
                  className="w-full p-3 bg-white/10 rounded-lg border border-white/20 focus:border-purple-400 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">
                  Temperature ({settings.temperature})
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.temperature}
                  onChange={(e) =>
                    setSettings((prev) => ({
                      ...prev,
                      temperature: parseFloat(e.target.value),
                    }))
                  }
                  className="w-full"
                />
              </div>
              <div>
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={settings.streamResponse}
                    onChange={(e) =>
                      setSettings((prev) => ({
                        ...prev,
                        streamResponse: e.target.checked,
                      }))
                    }
                    className="rounded"
                  />
                  <span className="text-sm font-medium">Stream Response</span>
                </label>
              </div>
            </div>
          </div>
        )}

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Section */}
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <Sparkles className="h-5 w-5 text-purple-400" />
                <span>Story Prompt</span>
              </h2>

              <div className="relative">
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Enter your story prompt here... (Ctrl/Cmd + Enter to generate)"
                  className="w-full h-32 p-4 bg-white/10 rounded-xl border border-white/20 focus:border-purple-400 focus:outline-none resize-none backdrop-blur-sm"
                />

                <button
                  onClick={generateStory}
                  disabled={!prompt.trim() || isGenerating}
                  className="absolute bottom-4 right-4 p-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded-lg transition-colors disabled:cursor-not-allowed"
                >
                  {isGenerating ? (
                    <Loader className="h-5 w-5 animate-spin" />
                  ) : (
                    <Send className="h-5 w-5" />
                  )}
                </button>
              </div>
            </div>

            {/* Example Prompts */}
            <div>
              <h3 className="text-lg font-medium mb-3">Example Prompts</h3>
              <div className="space-y-2">
                {examplePrompts.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setPrompt(example)}
                    className="w-full text-left p-3 bg-white/5 hover:bg-white/10 rounded-lg text-sm transition-colors border border-white/10"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Output Section */}
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold flex items-center space-x-2">
                <BookOpen className="h-5 w-5 text-purple-400" />
                <span>Generated Story</span>
              </h2>

              {story && (
                <button
                  onClick={downloadStory}
                  className="flex items-center space-x-2 px-3 py-1 bg-white/10 hover:bg-white/20 rounded-lg text-sm transition-colors"
                >
                  <Download className="h-4 w-4" />
                  <span>Download</span>
                </button>
              )}
            </div>

            <div
              ref={storyRef}
              className="h-96 p-4 bg-white/10 rounded-xl border border-white/20 overflow-y-auto backdrop-blur-sm"
            >
              {story ? (
                <div className="whitespace-pre-wrap text-gray-100 leading-relaxed">
                  {story}
                  {isGenerating && (
                    <span className="inline-block w-2 h-5 bg-purple-400 animate-pulse ml-1" />
                  )}
                </div>
              ) : (
                <div className="text-gray-400 text-center mt-8">
                  <BookOpen className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Your generated story will appear here...</p>
                  <p className="text-sm mt-2">
                    Enter a prompt and click generate to start!
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-400">
          <p className="text-sm">
            Powered by your optimized Transformer model • Built with React +
            FastAPI • Real-time streaming enabled
          </p>
        </div>
      </div>
    </div>
  );
};

export default StoryGenerator;
