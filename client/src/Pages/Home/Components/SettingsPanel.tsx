import React from "react";
import { GenerationSettings } from "../../../../../common/frontend-types";

// Settings Panel Component
interface SettingsPanelProps {
  settings: GenerationSettings;
  setSettings: React.Dispatch<React.SetStateAction<GenerationSettings>>;
  isVisible: boolean;
}

const SettingsPanel = ({
  settings,
  setSettings,
  isVisible,
}: SettingsPanelProps) => {
  if (!isVisible) return null;

  return (
    <div className="mb-8 p-6 bg-white/10 rounded-xl backdrop-blur-sm">
      <h3 className="text-lg font-semibold mb-4">Generation Settings</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium mb-2">Max Tokens</label>
          <input
            type="number"
            min="50"
            max="500"
            value={settings.maxTokens}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
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
            onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
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
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
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
  );
};

export default SettingsPanel;
