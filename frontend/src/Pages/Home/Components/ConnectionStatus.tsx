import React from "react";
import type { ConnectionStatus } from "../Common/types";

interface ConnectionStatusProps {
  status: ConnectionStatus;
}

const ConnectionStatusIndicator = ({ status }: ConnectionStatusProps) => (
    <div
      className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm ${
        status === "connected"
          ? "bg-green-500/20 text-green-400"
          : status === "error"
          ? "bg-red-500/20 text-red-400"
          : "bg-yellow-500/20 text-yellow-400"
      }`}
    >
      <div
        className={`w-2 h-2 rounded-full ${
          status === "connected"
            ? "bg-green-400"
            : status === "error"
            ? "bg-red-400"
            : "bg-yellow-400"
        }`}
      />
      {status}
    </div>
  );

export default ConnectionStatusIndicator;
