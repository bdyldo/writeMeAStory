import { useState, useRef, useEffect } from "react";
import Header from "./Components/Header";
import SettingsPanel from "./Components/SettingsPanel";
import PromptInput from "./Components/PromptInput";
import ExamplePrompts from "./Components/ExamplePrompts";
import StoryOutput from "./Components/StoryOutput";
import Footer from "./Components/Footer";
import useWebSocket from "./Hooks/useWebSocket";
import type { GenerationSettings } from "./Common/types";

const HomePage = () => {
  // State management
  const [prompt, setPrompt] = useState<string>("");
  const [story, setStory] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [settings, setSettings] = useState<GenerationSettings>({
    maxTokens: 200,
    temperature: 0.7,
    streamResponse: true,
  });
  const [showSettings, setShowSettings] = useState<boolean>(false);

  // Refs and hooks
  const storyRef = useRef<HTMLDivElement>(null);
  const { connectionStatus, wsRef } = useWebSocket();

  // Auto-scroll to bottom of story
  useEffect(() => {
    if (storyRef.current) {
      storyRef.current.scrollTop = storyRef.current.scrollHeight;
    }
  }, [story]);

  // WebSocket message handling
  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.onmessage = (event: MessageEvent) => {
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
    }
  }, [wsRef.current]);

  // Event handlers
  const handleGenerateStory = async (): Promise<void> => {
    if (!prompt.trim() || isGenerating) return;

    setIsGenerating(true);
    setStory("");

    if (
      settings.streamResponse &&
      wsRef.current?.readyState === WebSocket.OPEN
    ) {
      // WebSocket streaming
      const request = {
        type: "generate",
        prompt: prompt.trim(),
        max_tokens: settings.maxTokens,
        temperature: settings.temperature,
      };
      wsRef.current.send(JSON.stringify(request));
    } else {
      // HTTP fallback
      try {
        const response = await fetch("http://localhost:8000/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
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

  const handleDownloadStory = (): void => {
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      <div className="container mx-auto px-4 py-6">
        {/* Header Section */}
        <Header
          connectionStatus={connectionStatus}
          showSettings={showSettings}
          setShowSettings={setShowSettings}
        />

        {/* Settings Panel (Conditional) */}
        <SettingsPanel
          settings={settings}
          setSettings={setSettings}
          isVisible={showSettings}
        />

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Input Section */}
          <div className="space-y-6">
            <PromptInput
              prompt={prompt}
              setPrompt={setPrompt}
              onGenerate={handleGenerateStory}
              isGenerating={isGenerating}
            />

            <ExamplePrompts onSelectPrompt={setPrompt} />
          </div>

          {/* Right Column - Output Section */}
          <div>
            <StoryOutput
              story={story}
              isGenerating={isGenerating}
              onDownload={handleDownloadStory}
              storyRef={storyRef}
            />
          </div>
        </div>

        {/* Footer */}
        <Footer />
      </div>
    </div>
  );
};

export default HomePage;
