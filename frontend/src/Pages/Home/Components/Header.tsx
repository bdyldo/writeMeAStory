import React from "react";
import { BookOpen, Settings } from "lucide-react";
import type { ConnectionStatus } from "../Common/types";
import ConnectionStatusIndicator from "./ConnectionStatus";

// Header Component
interface HeaderProps {
  connectionStatus: ConnectionStatus;
  showSettings: boolean;
  setShowSettings: (show: boolean) => void;
}

const Header: React.FC<HeaderProps> = ({
  connectionStatus,
  showSettings,
  setShowSettings,
}) => (
  <div className="flex items-center justify-between mb-8">
    <div className="flex items-center space-x-3">
      <BookOpen className="h-8 w-8 text-purple-400" />
      <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
        AI Story Generator
      </h1>
    </div>

    <div className="flex items-center space-x-4">
      <ConnectionStatusIndicator status={connectionStatus} />
      <button
        onClick={() => setShowSettings(!showSettings)}
        className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
      >
        <Settings className="h-5 w-5" />
      </button>
    </div>
  </div>
);

export default Header;
