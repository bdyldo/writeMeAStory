import React from 'react'

interface ExamplePromptsProps {
    onSelectPrompt: (prompt: string) => void;
  }
  
  const ExamplePrompts: React.FC<ExamplePromptsProps> = ({ onSelectPrompt }) => {
    const examplePrompts: string[] = [
      "Once upon a time in a distant galaxy,",
      "The old lighthouse keeper discovered something unusual in the fog that night:",
      "Detective Sarah Chen had seen many strange cases, but this one was different:",
      "In the year 2157, humanity's first time traveler returned with urgent news:",
      "The ancient book glowed softly as Maya opened it for the first time:",
    ];
  
    return (
      <div>
        <h3 className="text-lg font-medium mb-3">Example Prompts</h3>
        <div className="space-y-2">
          {examplePrompts.map((example, index) => (
            <button
              key={index}
              onClick={() => onSelectPrompt(example)}
              className="w-full text-left p-3 bg-white/5 hover:bg-white/10 rounded-lg text-sm transition-colors border border-white/10"
            >
              {example}
            </button>
          ))}
        </div>
      </div>
    );
  };

export default ExamplePrompts