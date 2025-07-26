import React from "react";
import { BookOpen, Download } from "lucide-react";

interface StoryOutputProps {
  story: string;
  isGenerating: boolean;
  onDownload: () => void;
  storyRef: React.RefObject<HTMLDivElement | null>;
}

const StoryOutput = ({
  story,
  isGenerating,
  onDownload,
  storyRef,
}: StoryOutputProps) => (
  <div>
    <div className="flex items-center justify-between mb-4">
      <h2 className="text-xl font-semibold flex items-center space-x-2">
        <BookOpen className="h-5 w-5 text-purple-400" />
        <span>Generated Story</span>
      </h2>

      {story && (
        <button
          onClick={onDownload}
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
        <EmptyState />
      )}
    </div>
  </div>
);

// Empty State Component
const EmptyState: React.FC = () => (
  <div className="text-gray-400 text-center mt-8">
    <BookOpen className="h-12 w-12 mx-auto mb-4 opacity-50" />
    <p>Your generated story will appear here...</p>
    <p className="text-sm mt-2">Enter a prompt and click generate to start!</p>
  </div>
);

export default StoryOutput;
