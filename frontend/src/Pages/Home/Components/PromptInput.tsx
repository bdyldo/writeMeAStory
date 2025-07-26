import React from 'react'
import { Sparkles, Send, Loader } from "lucide-react";

interface PromptInputProps {
  prompt: string;
  setPrompt: (prompt: string) => void;
  onGenerate: () => void;
  isGenerating: boolean;
}

const PromptInput = ({ prompt, setPrompt, onGenerate, isGenerating }: PromptInputProps) => {
  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      onGenerate();
    }
  };

  return (
    <div>
      <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
        <Sparkles className="h-5 w-5 text-purple-400" />
        <span>Story Prompt</span>
      </h2>

      <div className="relative">
        <textarea
          value={prompt}
          onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setPrompt(e.target.value)}
          onKeyDown={handleKeyPress}
          placeholder="Enter your story prompt here... (Ctrl/Cmd + Enter to generate)"
          className="w-full h-32 p-4 bg-white/10 rounded-xl border border-white/20 focus:border-purple-400 focus:outline-none resize-none backdrop-blur-sm"
        />

        <button
          onClick={onGenerate}
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
  );
};

export default PromptInput